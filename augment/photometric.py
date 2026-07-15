"""
photometric.py
================

This module provides functions for applying photometric
augmentations to images. Photometric augmentations randomly
perturb image colour properties such as brightness, contrast,
saturation and hue. These transforms are commonly used to
increase robustness to lighting changes when training optical
flow networks. The implementation here follows the
augmentation strategy used in RAFT [1], but is generalised
to support an arbitrary number of images.

The main entry point is :func:`random_color_jitter_triplet`
which operates on three CHW tensors and applies the same
random colour jitter to all three images (symmetric mode) or
independent jitters to each image (asymmetric mode). The
probability of using asymmetric jitter is controlled via the
``asymmetric_prob`` argument. When asymmetric jitter is not
used the function concatenates the images along the height
dimension, applies a single jitter to the stack and then
splits the result back into individual images. This ensures
consistent colour changes across the sequence.

Example::

    from augment.photometric import random_color_jitter_triplet
    I1_aug, I2_aug, I3_aug = random_color_jitter_triplet(I1, I2, I3)

Notes
-----
* The images must be PyTorch tensors of shape ``(C, H, W)``
  with values in the range ``[0, 1]``. A ``ValueError`` is
  raised for other shapes.
* Colour jittering is implemented using ``torchvision``'s
  :class:`ColorJitter` transform; therefore ``torchvision``
  must be installed. If ``ColorJitter`` cannot be imported an
  ``ImportError`` will be raised.

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

try:
    # Use torchvision's colour jitter implementation if available
    from torchvision.transforms import ColorJitter
    from torchvision.transforms.functional import to_pil_image, to_tensor
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "The photometric augmentation module requires torchvision to be installed."
    ) from exc


def _validate_image(img: torch.Tensor) -> None:
    """Validate that ``img`` is a 3D tensor in CHW format with values in [0,1]."""
    if not torch.is_tensor(img):
        raise TypeError(f"Expected img to be a torch.Tensor, got {type(img)}")
    if img.ndim != 3:
        raise ValueError(f"Expected 3D tensor (C,H,W), got shape {tuple(img.shape)}")
    if img.max() > 1.0 or img.min() < 0.0:
        raise ValueError("Input image tensor must have values in [0,1]")


def _apply_jitter(img: torch.Tensor, jitter: ColorJitter) -> torch.Tensor:
    """Apply a :class:`ColorJitter` transform to a single image tensor.

    Parameters
    ----------
    img: torch.Tensor
        A 3‑channel image tensor of shape (C, H, W) in [0, 1].
    jitter: ColorJitter
        A configured colour jitter transform.

    Returns
    -------
    torch.Tensor
        The colour‑jittered image as a tensor with the same shape and
        value range as the input.
    """
    _validate_image(img)
    # Convert to PIL for ColorJitter then back to tensor. PIL expects HWC
    pil_img = to_pil_image(img)
    jittered = jitter(pil_img)
    # Convert back to tensor in [0,1]
    return to_tensor(jittered)


def random_color_jitter_triplet(
    I1: torch.Tensor,
    I2: torch.Tensor,
    I3: torch.Tensor,
    *,
    brightness: float = 0.4,
    contrast: float = 0.4,
    saturation: float = 0.4,
    hue: float = 0.5 / 3.141592653589793,
    asymmetric_prob: float = 0,
    apply_prob: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply random colour jitter to a triplet of images.

    This function returns augmented versions of the input images. The
    augmentations are applied with probability ``apply_prob``; if the
    augmentation is skipped the original inputs are returned unchanged.
    When applied, a :class:`ColorJitter` transform is configured
    according to the supplied parameters. With probability
    ``asymmetric_prob`` the jitter is applied independently to each
    image (asymmetric augmentation). Otherwise the same jitter is
    applied to all three images (symmetric augmentation). Symmetric
    augmentation is implemented by stacking the images along the
    height dimension, applying the jitter once to the concatenated
    image and then splitting the result back into individual images.

    Parameters
    ----------
    I1, I2, I3: torch.Tensor
        Input images of shape ``(C, H, W)`` in the range ``[0, 1]``.
    brightness, contrast, saturation, hue: float
        Parameters passed to :class:`ColorJitter`. See PyTorch
        documentation for details.
    asymmetric_prob: float
        Probability of applying different jitter parameters to each image.
    apply_prob: float
        Overall probability of performing colour jitter. If a random
        number in ``[0,1)`` exceeds this value then no augmentation is
        applied and the inputs are returned as‑is.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        The possibly augmented images ``(I1_aug, I2_aug, I3_aug)``.
    """
    # Validate inputs
    for img in (I1, I2, I3):
        _validate_image(img)
    # Decide whether to apply augmentation
    if random.random() >= apply_prob:
        return I1, I2, I3

    # Configure jitter
    jitter = ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )

    # Determine asymmetric or symmetric jitter
    if random.random() < asymmetric_prob:
        # Asymmetric: each image gets its own random jitter
        I1_aug = _apply_jitter(I1, jitter)
        I2_aug = _apply_jitter(I2, jitter)
        I3_aug = _apply_jitter(I3, jitter)
    else:
        # Symmetric: stack along height and apply once
        # Convert tensors to PIL images stacked vertically
        # Stack as H*3 x W
        C, H, W = I1.shape
        stacked = torch.cat([I1, I2, I3], dim=1)  # CHW -> C(H*3)W
        pil_img = to_pil_image(stacked)
        jittered = jitter(pil_img)
        jittered_t = to_tensor(jittered)
        # Split back
        I1_aug = jittered_t[:, 0:H, :]
        I2_aug = jittered_t[:, H : 2 * H, :]
        I3_aug = jittered_t[:, 2 * H :, :]

    return I1_aug, I2_aug, I3_aug
