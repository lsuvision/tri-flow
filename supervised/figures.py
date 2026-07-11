""" Code to create flow visualizations of different datasets in batches """

import sys
sys.path.append('core')
import argparse
import cv2
import numpy as np
import torch
from PIL import Image
from raft import RAFT
from utils import flow_viz
import datasets
from utils.utils import InputPadder




DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo, img_name = "optical_flow.png"):
    img = img[0].permute(1,2,0).cpu().numpy()

    if flo.ndim == 3:
        flo = flo.permute(1,2,0).cpu().numpy()
    else:
        flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    img_flo = flow_viz.flow_to_image(flo)

    # save the flow visualization
    img_bgr = cv2.cvtColor(img_flo, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_name , img_bgr)


@torch.no_grad()
def chairs_flow(model_raft, model_ours, args, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split and create flow visualizations """
    model_raft.eval()
    model_ours.eval()

    val_dataset = datasets.FlyingChairs(split='validation')

    try:
        high = int(args.pairs)
    except Exception as e:
        print(e)
        print("Defaulting number of pairs to 5")
        high = 5
    for val_id in range(high):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr_raft = model_raft(image1, image2, iters=iters, test_mode=True)
        _, flow_pr_ours = model_ours(image1, image2, iters=iters, test_mode=True)

        epe_raft = torch.sum((flow_pr_raft[0].cpu() - flow_gt)**2, dim=0).sqrt()
        avg_epe_raft = epe_raft.view(-1).mean().item()

        epe_ours = torch.sum((flow_pr_ours[0].cpu() - flow_gt)**2, dim=0).sqrt()
        avg_epe_ours = epe_ours.view(-1).mean().item()

        sequence_name  = val_dataset.image_list[val_id]
        img1_name = sequence_name[0].split('/')[-1]
        print(f"our model name: {args.ours}, sequence: {sequence_name}, EPE Raft: {avg_epe_raft}, EPE Ours: {avg_epe_ours}")

        # save original image
        img1 = image1[0].permute(1, 2, 0).cpu().numpy()

        cv2.imwrite(f"original_image_{img1_name}", img1[..., [2, 1, 0]])
        # compute flow visualizations for both models as well as ground truth and save
        viz(image1, flow_pr_raft, img_name=f"optical_flow_raft_{img1_name}")
        viz(image1, flow_pr_ours, img_name=f"optical_flow_ours_{img1_name}")
        viz(image1, flow_gt, img_name=f"optical_flow_gt_{img1_name}")

    return 


@torch.no_grad()
def things_flow(model_raft, model_ours, args, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split and create flow visualizations """
    model_raft.eval()
    model_ours.eval()

    val_dataset = datasets.FlyingThings3D(split='TEST')

    try:
        high = int(args.pairs)
    except Exception as e:
        print(e)
        print("Defaulting number of pairs to 5")
        high = 5
    for val_id in range(high):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        _, flow_pr_raft = model_raft(image1, image2, iters=iters, test_mode=True)
        flow_raft = padder.unpad(flow_pr_raft[0]).cpu()

        _, flow_pr_ours = model_ours(image1, image2, iters=iters, test_mode=True)
        flow_ours = padder.unpad(flow_pr_ours[0]).cpu()

        epe_raft = torch.sum((flow_raft[0].cpu() - flow_gt)**2, dim=0).sqrt()
        avg_epe_raft = epe_raft.view(-1).mean().item()

        epe_ours = torch.sum((flow_ours[0].cpu() - flow_gt)**2, dim=0).sqrt()
        avg_epe_ours = epe_ours.view(-1).mean().item()

        sequence_name  = val_dataset.image_list[val_id]
        img1_name = sequence_name[0].split('/')[-1]
        print(f"our model name: {args.ours}, sequence: {sequence_name}, EPE Raft: {avg_epe_raft}, EPE Ours: {avg_epe_ours}")

        # save original image
        img1 = image1[0].permute(1, 2, 0).cpu().numpy()

        cv2.imwrite(f"original_image_{img1_name}", img1[..., [2, 1, 0]])
        # compute flow visualizations for both models as well as ground truth and save
        viz(image1, flow_pr_raft, img_name=f"optical_flow_raft_{img1_name}")
        viz(image1, flow_pr_ours, img_name=f"optical_flow_ours_{img1_name}")
        viz(image1, flow_gt, img_name=f"optical_flow_gt_{img1_name}")

    return 


@torch.no_grad()
def sintel_flow(model_raft, model_ours, args, iters=32):
    """ Peform validation using the Sintel (train) split and create flow visualizations"""
    model_raft.eval()
    model_ours.eval()


    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            # get flow from raft
            flow_low, flow_pr_raft = model_raft(image1, image2, iters=iters, test_mode=True)
            flow_raft = padder.unpad(flow_pr_raft[0]).cpu()

            #  get flow from our model
            flow_low, flow_pr_ours = model_ours(image1, image2, iters=iters, test_mode=True)
            flow_ours = padder.unpad(flow_pr_ours[0]).cpu()
            
            # epe for raft 
            epe = torch.sum((flow_raft - flow_gt)**2, dim=0).sqrt()
            epe = epe.view(-1)

            # get epe for the flow
            avg_epe_raft = epe.mean().item()

            # epe for our model 
            epe = torch.sum((flow_ours - flow_gt)**2, dim=0).sqrt()
            epe = epe.view(-1)
            # get epe for the flow
            avg_epe_ours = epe.mean().item()

            # print our epes and the name of the sequence
            sequence_name  = val_dataset.image_list[val_id]
            img1_name = sequence_name[0].split('/')[-1]
            print(f"our model name: {args.ours}, sequence: {sequence_name}, EPE Raft: {avg_epe_raft}, EPE Ours: {avg_epe_ours}")

            # save original image
            img1 = image1[0].permute(1, 2, 0).cpu().numpy()

            cv2.imwrite(f"original_image_{img1_name}", img1[..., [2, 1, 0]])
            # compute flow visualizations for both models as well as ground truth and save
            viz(image1, flow_raft, img_name=f"optical_flow_raft_{img1_name}")
            viz(image1, flow_ours, img_name=f"optical_flow_ours_{img1_name}")
            viz(image1, flow_gt, img_name=f"optical_flow_gt_{img1_name}")
    return


@torch.no_grad()
def kitti_flow(model_raft, model_ours, args, iters=24):
    """ Create figures for the KITTI-2015 (train) split """
    model_raft.eval()
    model_ours.eval()
    val_dataset = datasets.KITTI(split='training')

    try:
        offset = int(args.offset) 
        high = offset + int(args.pairs)
    except Exception as e:
        print(e)
        print("Defaulting number of pairs to 5")
        offset = 0
        high = offset + 5


    for val_id in range(offset, high):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        # get flow from raft
        flow_low, flow_pr_raft = model_raft(image1, image2, iters=iters, test_mode=True)
        flow_raft = padder.unpad(flow_pr_raft[0]).cpu()

        #  get flow from our model
        flow_low, flow_pr_ours = model_ours(image1, image2, iters=iters, test_mode=True)
        flow_ours = padder.unpad(flow_pr_ours[0]).cpu()

        # epe for raft 
        epe = torch.sum((flow_raft - flow_gt)**2, dim=0).sqrt()
        epe = epe.view(-1)
        val = valid_gt.view(-1) >= 0.5
        # get epe for the flow
        avg_epe_raft = epe[val].mean().item()

        # epe for our model 
        epe = torch.sum((flow_ours - flow_gt)**2, dim=0).sqrt()
        epe = epe.view(-1)
        # get epe for the flow
        avg_epe_ours = epe[val].mean().item()

        # print our epes and the name of the sequence
        sequence_name  = val_dataset.image_list[val_id]
        img1_name = sequence_name[0].split('/') 
        img1_name = img1_name[-2] + img1_name[-1]
        print(f"our model name: {args.ours}, sequence: {sequence_name}, EPE Raft: {avg_epe_raft}, EPE Ours: {avg_epe_ours}")

        # save original image
        img1 = image1[0].permute(1, 2, 0).cpu().numpy()

        cv2.imwrite(f"original_image_{img1_name}", img1[..., [2, 1, 0]])
        # compute flow visualizations for both models as well as ground truth and save
        viz(image1, flow_raft, img_name=f"optical_flow_raft_{img1_name}")
        viz(image1, flow_ours, img_name=f"optical_flow_ours_{img1_name}")
        viz(image1, flow_gt, img_name=f"optical_flow_gt_{img1_name}")

    return


@torch.no_grad()
def hd1k_flow(model_raft, model_ours, args, iters=24):
    """ take 2 models, calculate epe, visualize and save the original image, model1 flow, model2 flow, and ground truth flow """
    model_raft.eval()
    model_ours.eval()
    val_dataset = datasets.HD1K()

    try:
        high = int(args.pairs)
    except Exception as e:
        print(e)
        print("Defaulting number of pairs to 5")
        high = 5


    for val_id in range(high):


        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        # get flow from raft
        flow_low, flow_pr_raft = model_raft(image1, image2, iters=iters, test_mode=True)
        flow_raft = padder.unpad(flow_pr_raft[0]).cpu()

        #  get flow from our model
        flow_low, flow_pr_ours = model_ours(image1, image2, iters=iters, test_mode=True)
        flow_ours = padder.unpad(flow_pr_ours[0]).cpu()
        
        # epe for raft 
        epe = torch.sum((flow_raft - flow_gt)**2, dim=0).sqrt()
        epe = epe.view(-1)
        val = valid_gt.view(-1) >= 0.5
        # get epe for the flow
        avg_epe_raft = epe[val].mean().item()

        # epe for our model 
        epe = torch.sum((flow_ours - flow_gt)**2, dim=0).sqrt()
        epe = epe.view(-1)
        # get epe for the flow
        avg_epe_ours = epe[val].mean().item()

        # print our epes and the name of the sequence
        sequence_name  = val_dataset.image_list[val_id]
        img1_name = sequence_name[0].split('/')[-1]
        print(f"our model name: {args.ours}, sequence: {sequence_name}, EPE Raft: {avg_epe_raft}, EPE Ours: {avg_epe_ours}")

        # save original image
        img1 = image1[0].permute(1, 2, 0).cpu().numpy()

        cv2.imwrite(f"original_image_{img1_name}", img1[..., [2, 1, 0]])
        # compute flow visualizations for both models as well as ground truth and save
        viz(image1, flow_raft, img_name=f"optical_flow_raft_{img1_name}")
        viz(image1, flow_ours, img_name=f"optical_flow_ours_{img1_name}")
        viz(image1, flow_gt, img_name=f"optical_flow_gt_{img1_name}")

    return 


if __name__ == '__main__':
    torch.set_num_threads(1) 

    parser = argparse.ArgumentParser()
    parser.add_argument('--raft', help="restore checkpoint for original raft")
    parser.add_argument('--ours', help="restore checkpoint for our model")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--pairs', help="number of image pairs to evaluate")
    parser.add_argument('--offset', help="integer offset to start evaluating images from")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    model_raft = torch.nn.DataParallel(RAFT(args))
    model_raft.load_state_dict(torch.load(args.raft))

    model_ours = torch.nn.DataParallel(RAFT(args))
    model_ours.load_state_dict(torch.load(args.ours))

    model_raft.cuda()
    model_raft.eval()

    model_ours.cuda()
    model_ours.eval()


    with torch.no_grad():
       if args.dataset == 'chairs':
           chairs_flow(model_raft.module, model_ours.module, args)

       elif args.dataset == 'things':
           things_flow(model_raft.module, model_ours.module, args)

       elif args.dataset == 'sintel':
           sintel_flow(model_raft.module, model_ours.module, args)

       elif args.dataset == 'kitti':
           kitti_flow(model_raft.module, model_ours.module, args)

       elif args.dataset == 'hd1k':
           hd1k_flow(model_raft.module, model_ours.module, args)
