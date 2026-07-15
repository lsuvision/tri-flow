"""
occlusion.py
============

Functions for inserting synthetic occlusions into images. Occlusion
augmentation helps train optical flow networks to handle missing
pixels and disocclusions by simulating regions where no valid
photometric correspondence exists. The implementation here
follows the occlusion augmentation used in RAFT [1] and
subsequent self‑supervised optical flow methods (e.g. SMURF,
ARFlow).

The :func:`random_occlusion_triplet` function operates on a
sequence of three images and inserts rectangular patches of
constant colour into the second image. The colour is set to the
mean colour of the image to minimise the effect of the occluded
region on global brightness statistics. Patches are inserted
multiple times with random sizes and positions drawn from a
specified range. The first and third images are returned
unchanged to preserve temporal consistency.

Example::

    from augment.occlusion import random_occlusion_triplet
    I1_o, I2_o, I3_o = random_occlusion_triplet(I1, I2, I3, prob=0.5)

Notes
-----
* All images must be PyTorch tensors of shape ``(C, H, W)`` with
  values in ``[0, 1]``. A ``ValueError`` is raised for invalid
  inputs.
* The occlusion is applied only to the second image (middle frame).

References
----------
[1] Zachary Teed and Jia Deng. "RAFT: Recurrent All‑Pairs Field
    Transforms for Optical Flow." In _European Conference on
    Computer Vision_, 2020.
"""

from __future__ import annotations

import random
from typing import Tuple

import torch


def _validate_image(img: torch.Tensor) -> None:
    """Ensure ``img`` is a valid CHW tensor in [0, 1]."""
    if not torch.is_tensor(img):
        raise TypeError(f"Expected img to be a torch.Tensor, got {type(img)}")
    if img.ndim != 3:
        raise ValueError(f"Expected 3D tensor (C,H,W), got shape {tuple(img.shape)}")
    if img.max() > 1.0 or img.min() < 0.0:
        raise ValueError("Input image tensor must have values in [0,1]")


def random_occlusion_triplet(
    I1: torch.Tensor,
    I2: torch.Tensor,
    I3: torch.Tensor,
    *,
    prob: float = 0.5,
    bounds: Tuple[int, int] = (50, 100),
    max_patches: Tuple[int, int] = (1, 2),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Insert rectangular occlusions into the second image of a triplet.

    With probability ``prob`` this function inserts one or more
    rectangular patches into ``I2``. The patches have width and
    height uniformly drawn from ``bounds``. The colour of each
    patch is set to the mean colour of ``I2``. The number of
    patches inserted is randomly chosen between ``max_patches[0]``
    and ``max_patches[1]`` inclusive. When the augmentation is not
    applied the input images are returned unchanged.

    Parameters
    ----------
    I1, I2, I3: torch.Tensor
        Input images of shape ``(C, H, W)`` in ``[0, 1]``.
    prob: float
        Probability of inserting occlusions. If a random number in
        ``[0,1)`` exceeds this value no occlusion is applied.
    bounds: Tuple[int,int]
        Minimum and maximum side length (in pixels) for each
        occlusion patch. Width and height are sampled
        independently from this range.
    max_patches: Tuple[int,int]
        The minimum and maximum number of patches to insert.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        The images ``(I1_o, I2_o, I3_o)`` with occlusions inserted
        into ``I2``. ``I1`` and ``I3`` are returned unchanged.
    """
    # Validate inputs
    for img in (I1, I2, I3):
        _validate_image(img)

    # Decide whether to apply augmentation
    if random.random() >= prob:
        return I1, I2, I3

    C, H, W = I2.shape
    I2_aug = I2.clone()
    # Compute mean colour (per channel) across image
    mean_colour = I2_aug.view(C, -1).mean(dim=1)
    num_patches = random.randint(max_patches[0], max_patches[1])
    for _ in range(num_patches):
        # Random top‑left corner
        x0 = random.randint(0, max(0, W - 1))
        y0 = random.randint(0, max(0, H - 1))
        # Random size
        dx = random.randint(bounds[0], bounds[1])
        dy = random.randint(bounds[0], bounds[1])
        # Clip to image boundaries
        x1 = min(W, x0 + dx)
        y1 = min(H, y0 + dy)
        # Fill patch with mean colour
        # Broadcasting mean_colour (C,) to (C, dy, dx)
        I2_aug[:, y0:y1, x0:x1] = mean_colour.view(C, 1, 1)

    return I1, I2_aug, I3
