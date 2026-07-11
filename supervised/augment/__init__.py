"""
Augmentations
=============

This subpackage implements simple geometric augmentations for images and
flow fields. At present only affine transforms composed of
translation, rotation and uniform scaling (TRS) are supported. When
applying an affine to an image we return the corresponding output‑to‑input
matrix suitable for use with ``torch.nn.functional.affine_grid`` and
``grid_sample``. For flow fields we provide an implementation of the
"Aug*" operator described in the accompanying paper which maps a flow
field under an affine augmentation of the target frame.

Public API:

* :class:`TRSRange` – a dataclass specifying the ranges from which
  translations, rotations and scales are sampled.
* :func:`sample_trs` – sample random TRS parameters per batch.
* :func:`build_affine_out2in` – build pixel‑coordinate affine matrices
  (output → input) for grid sampling.
* :func:`apply_affine` – apply an affine to an image tensor.
* :func:`aug_star_flow` – apply the Aug* operator to a flow field.

These functions operate on batched tensors and support differentiable
operations, making them suitable for use in the training loop.
"""

from .trs import (
    TRSRange,
    sample_trs,
    build_affine_out2in,
    apply_affine,
    aug_star_flow,
    invert_affine_2x3,
    apply_affine_to_coords,
    _base_grid,
)

# Import photometric and occlusion augmentation utilities. These provide
# additional data augmentation capabilities beyond simple affine
# transforms. They are exposed at the package level for convenience.
from .photometric import random_color_jitter_triplet  # noqa:F401
from .occlusion import random_occlusion_triplet  # noqa:F401
from .utils import random_horizontal_flip_triplet, random_reverse_triplet  # noqa:F401

__all__ = [
    # Affine augmentation API
    "TRSRange",
    "sample_trs",
    "build_affine_out2in",
    "apply_affine",
    "aug_star_flow",
    # Additional affine helpers
    "invert_affine_2x3",
    "apply_affine_to_coords",
    "_base_grid",
    # Photometric augmentation API
    "random_color_jitter_triplet",
    # Occlusion augmentation API
    "random_occlusion_triplet",
    # Basic augmentation utilities
    "random_horizontal_flip_triplet",
    "random_reverse_triplet",
]