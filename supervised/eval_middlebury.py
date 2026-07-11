import sys
sys.path.append('core')

import numpy as np
import cv2
import os
import argparse
import torch

from raft import RAFT
from utils.utils import InputPadder


def read_middlebury_flow(flow_path):
    """
    Reads optical flow from Middlebury .flo file
    Returns: flow (H, W, 2) float32 numpy array
    """
    with open(flow_path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)[0]
        if magic != 202021.25:
            raise ValueError("Invalid .flo file")

        width = np.fromfile(f, np.int32, count=1)[0]
        height = np.fromfile(f, np.int32, count=1)[0]

        data = np.fromfile(f, np.float32, count=height * width * 2)

    flow = np.reshape(data, (height, width, 2))
    return flow


def load_sample(scenes_path, gt_path, curr_scene):
    """
    Loads frame10.png, frame11.png and flow.flo from a folder.
    Returns: img1, img2, flow
    """
    img1_path = os.path.join(scenes_path, curr_scene, "frame10.png")
    img2_path = os.path.join(scenes_path, curr_scene, "frame11.png")
    flow_path = os.path.join(gt_path, curr_scene, "flow10.flo")

    # Load images (BGR -> RGB)
    img1 = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2RGB)

    flow = read_middlebury_flow(flow_path)

    return img1, img2, flow





def compute_epe(model, model_name, dataset_path, iters = 24): 
    model.eval()
    epe_list = []
    
    gt_path = "other-gt-flow"
    gt_path = os.path.join(dataset_path, gt_path)

    scenes_path = "other-data"
    scenes_path = os.path.join(dataset_path, scenes_path)
    
    gt_folders = os.scandir(gt_path)
    gt_folders = [entry.name for entry in gt_folders]

    with torch.no_grad():
        for entry in os.scandir(scenes_path):
            if entry.name not in gt_folders:
                continue

            image1, image2, gt_flow = load_sample(scenes_path, gt_path, entry.name)

            image1 = torch.from_numpy(image1).permute(2, 0, 1).float() 
            image2 = torch.from_numpy(image2).permute(2, 0, 1).float() 
            gt_flow = torch.from_numpy(gt_flow).permute(2, 0, 1).float()  


            valid = (torch.abs(gt_flow[0]) < 1e9) & (torch.abs(gt_flow[1]) < 1e9)

            # add an extra dimension for the batch
            image1 = image1.unsqueeze(0)
            image2 = image2.unsqueeze(0)

            image1 = image1.cuda()
            image2 = image2.cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()


            epe = torch.sum((flow - gt_flow)**2, dim=0).sqrt()
            epe = epe.view(-1)
            valid = valid.view(-1)
            epe = epe[valid].mean().item()
            epe_list.append(epe)
            epe_list.append(epe)


             
    epe_list = np.array(epe_list)
    epe = np.mean(epe_list)
    print(f"Checkpoint: {model_name},  EPE: {epe}")



# Example usage
if __name__ == "__main__":

    parser  = argparse.ArgumentParser()
    parser.add_argument("--model", help="Model to evaluate on Middlebury")
    parser.add_argument("--dataset", help="Dataset to evaluate")
    parser.add_argument("--small", action="store_true", help="Use small model")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed_precision")
    parser.add_argument("--alternate_corr", action="store_true", help="Use efficient correlation implementation")
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    Middlebury_path = "./datasets/MiddleburyGT"

    compute_epe(model.module, args.model, Middlebury_path)

