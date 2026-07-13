# RAFT with Triangular Consistency

[[`arXiv`](https://arxiv.org/abs/2606.19938)]

## Introduction

This folder contains our implementation of triangular consistency into the [RAFT](https://github.com/princeton-vl/raft) supervised optical flow model (Note: Augmentation Consistency is the only consistency method applied for this experiment).

## Training Script

The main changes occur in the script [train.py](train.py). While it closely follows the original script, there are some key changes in the `train`, `sequence_loss` functions and entry point of the script.

In the `train` function, we use the following code whose purpose is to obtain an augmented frame, as well as to obtain the composed flow:

``` Python
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

```

The above computations are grouped together in a function ________________ in the script [tri_flow.py](../tri_flow.py)_________ for simplicity. Please refer to the root [README](../README.md) for general instructions on how to apply augmentation consistency to your optical flow method.

With the augmented image, we can predict the flow from `frame_0` to the augmented `frame_1`.
Then we pass these predictions, along with the analytically augmented flow to RAFT's `sequence_loss` function. We modified the `sequence_loss` function to accept an exra argument, which is the validity mask resulting from augmentation whose purpose is to mask out from the loss pixels that land outside the image bounds after the transformation. If this validity mask is provided then both this and the original dataset-specific validity mask will be applied. Please note that we do not mask out the pixels that land outside of the image bounds for our experiments.

## Augmentation Parameters

We also expand the number of command line arguments the script can take to control the range of the augmentation parameters.
The command line arguments `aug_tx`, `aug_ty`, `aug_rot`, `aug_smin`, `aug_smax` can be specified as follows:

``` Bash
python -u train.py \
    --name raft-sintel \
    --stage sintel \
    --validation sintel \
    --restore_ckpt checkpoints/raft-things.pth \
    --gpus 0 1 \
    --num_steps 100000 \
    --batch_size 6 \
    --lr 0.000125 \
    --image_size 368 768 \
    --wdecay 0.00001 \
    --gamma=0.85 \
    --aug_tx 2.0 \
    --aug_ty 2.0 \
    --aug_rot 2.0 \
    --aug_smin 0.95 \
    --aug_smax 1.0
```

If the augmentation parameters are not specified, they default to values that correspond to performing no augmentation.

For more details and documentation please refer to the script [trs.py](augment/trs.py) for augmentations, or our [paper](https://arxiv.org/abs/2606.19938).







