"""
utils.py
========

Utility functions for simple augmentation operations on sequences of
images. These helpers support operations such as horizontal
flipping and reversing the temporal order of an image triplet.
These transformations are useful for increasing the diversity of
training data in self‑supervised optical flow learning.

The functions here operate on image triplets represented as
PyTorch tensors in CHW format. They are designed to be
composable and side‑effect free: each function returns new
tensors and does not modify the inputs in place.

Example::

    from augment.utils import random_horizontal_flip_triplet, random_reverse_triplet
    I1_f, I2_f, I3_f = random_horizontal_flip_triplet(I1, I2, I3, prob=0.5)
    I1_r, I2_r, I3_r = random_reverse_triplet(I1, I2, I3, prob=0.5)
"""

from __future__ import annotations

import random
from typing import Tuple

import torch


def _validate_image(img: torch.Tensor) -> None:
    """Check that ``img`` is a 3D tensor (C,H,W)."""
    if not torch.is_tensor(img):
        raise TypeError(f"Expected a torch.Tensor, got {type(img)}")
    if img.ndim != 3:
        raise ValueError(f"Expected a 3D tensor, got shape {tuple(img.shape)}")


def random_horizontal_flip_triplet(
    I1: torch.Tensor, I2: torch.Tensor, I3: torch.Tensor, *, prob: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomly flip all three images horizontally.

    With probability ``prob`` this function reverses the width
    dimension of each input image. Otherwise the inputs are
    returned unchanged.

    Parameters
    ----------
    I1, I2, I3: torch.Tensor
        Input images of shape ``(C, H, W)``.
    prob: float
        Probability of applying the flip.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        The (possibly flipped) images.
    """
    for img in (I1, I2, I3):
        _validate_image(img)

    if random.random() >= prob:
        return I1, I2, I3

    # Flip along width (last dimension)
    return I1.flip(dims=[2]), I2.flip(dims=[2]), I3.flip(dims=[2])


def random_reverse_triplet(
    I1: torch.Tensor, I2: torch.Tensor, I3: torch.Tensor, *, prob: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomly reverse the temporal order of the triplet.

    With probability ``prob`` the order of the images is reversed
    from ``(I1, I2, I3)`` to ``(I3, I2, I1)``. Otherwise the input
    order is preserved.

    Parameters
    ----------
    I1, I2, I3: torch.Tensor
        Input images of shape ``(C, H, W)``.
    prob: float
        Probability of reversing the order.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        The images in either original or reversed order.
    """
    for img in (I1, I2, I3):
        _validate_image(img)

    if random.random() >= prob:
        return I1, I2, I3
    # Reverse order
    return I3, I2, I1
