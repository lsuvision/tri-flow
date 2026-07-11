from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from raft import RAFT
import evaluate
import datasets

from torch.utils.tensorboard import SummaryWriter
from augment import TRSRange, sample_trs, build_affine_out2in, apply_affine, invert_affine_2x3, apply_affine_to_coords, _base_grid  # type: ignore

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000

def warp(x, flow):
    """Warps image x (B, C, H, W) using flow (B, 2, H, W) via bilinear sampling."""
    B, C, H, W = x.size()
    # Mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing='ij')
    grid = torch.stack((grid_x, grid_y), 2).float().unsqueeze(0).expand(B, -1, -1, -1)
    
    vgrid = grid + flow.permute(0, 2, 3, 1)
    # Scale grid to [-1, 1] for grid_sample
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    
    # Grid sample expects [-1, 1] coordinates
    return F.grid_sample(x, torch.stack((vgrid_x, vgrid_y), 3), padding_mode='border', align_corners=True)

def transform_flow_with_matrix(f01: torch.Tensor, A_out2in: torch.Tensor) -> torch.Tensor:
    """
    Transforms optical flow f01 (from I0 to I1) to match the affine-transformed I1.

    f01: Tensor of shape (B, 2, H, W) : flow from I0 to I1
    A_out2in: Tensor of shape (B, 2, 3) : affine matrices from augmented I1 back to original I1

    Returns: flow from I0 to affine(I1), with invalid areas zeroed out.
    """

    B, _, H, W = f01.shape

    # Invert the A_out2in to get input <- output (I1 <- I1_aug)
    A_in2out = invert_affine_2x3(A_out2in)

    # Base pixel coordinate grid
    x = _base_grid(B, H, W, f01.device)  # shape (B, 2, H, W)

    # Original target positions (y = x + f01)
    y = x + f01

    # Transform target positions to augmented space
    y_prime = apply_affine_to_coords(y, A_in2out)

    # New flow is from x to y_prime
    f01_prime = y_prime - x

    # Optional: mask invalid pixels outside image bounds
    valid_x = (y_prime[:, 0] >= 0.0) & (y_prime[:, 0] <= W - 1)
    valid_y = (y_prime[:, 1] >= 0.0) & (y_prime[:, 1] <= H - 1)
    valid_mask = (valid_x & valid_y).float()
    
    
    return f01_prime, valid_mask

def sequence_loss(flow_preds, flow_gt, valid, valid_aug, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exclude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    if valid_aug is not None:
        valid = (valid >= 0.5) & (mag < max_flow) & (valid_aug >= 0.5)
    else:
        valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()

beta = 1.

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.cuda()
    model.train()

    if args.stage != 'chairs':
        model.module.freeze_bn()

    # changed training from  "C+T+K+S+H" to  "C+T+K/S" 
    train_loader = datasets.fetch_dataloader(args, TRAIN_DS='C+T+K/S')
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    VAL_FREQ = 5000
    add_noise = True

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            data_blob = [x.cuda() for x in data_blob]
            
            if len(data_blob) == 4:
                image1, image2, flow, valid = data_blob
            else:
                image1, image2, flow = data_blob
                valid = torch.ones_like(flow[:, 0], device=flow.device)


            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            trs = TRSRange(
                tx=(-args.aug_tx, args.aug_tx),
                ty=(-args.aug_ty, args.aug_ty),
                rot=(-args.aug_rot, args.aug_rot),
                scale=(args.aug_smin, args.aug_smax),
            )

            B, _, H, W = image1.shape
            trs_params = sample_trs(B, trs, device)
            A_out2in = build_affine_out2in(*trs_params, H, W)

            image2_aug = apply_affine(image2, A_out2in)

            flow_aug, valid_aug = transform_flow_with_matrix(flow, A_out2in)
            flow_predictions = model(image1, image2_aug, iters=args.iters)            
            flow_predictions_original = model(image1, image2, iters=args.iters)            
            
            
            
            loss_augmentation, metrics = sequence_loss(flow_predictions, flow_aug, valid, valid_aug, args.gamma)
            loss_original, metrics_original = sequence_loss(flow_predictions_original, flow, valid, None, args.gamma)
            loss = (loss_original + beta * loss_augmentation) / (1 + beta)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics_original)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
              PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
              torch.save(model.state_dict(), PATH)
  
              results = {}
              for val_dataset in args.validation:
                  if val_dataset == 'chairs':
                      results.update(evaluate.validate_chairs(model.module))
                  elif val_dataset == 'sintel':
                      results.update(evaluate.validate_sintel(model.module))
                  elif val_dataset == 'kitti':
                      results.update(evaluate.validate_kitti(model.module))

              logger.write_dict(results)
            
              model.train()
              if args.stage != 'chairs':
                  model.module.freeze_bn()
          
            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    # Augmentation ranges
    parser.add_argument("--aug_tx", type=float, default=0.0, help="Max translation in pixels in x direction")
    parser.add_argument("--aug_ty", type=float, default=0.0, help="Max translation in pixels in y direction")
    parser.add_argument("--aug_rot", type=float, default=0.0, help="Max rotation in degrees")
    parser.add_argument("--aug_smin", type=float, default=1.0, help="Minimum scale factor")
    parser.add_argument("--aug_smax", type=float, default=1.0, help="Maximum scale factor")
    args = parser.parse_args()

 #   torch.manual_seed(1234)
 #   np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)
