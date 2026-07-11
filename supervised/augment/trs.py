"""
trs.py
======

Implementation of simple TRS (translation, rotation, scale) affine
augmentations. These transforms operate on batched image tensors and
flow fields. The key idea is to sample a random affine for each
element in the batch and then construct the appropriate matrices for
warping images and flows.

*Translations* are specified in pixels and are applied directly.

*Rotations* are specified in degrees and are applied around the
centre of the image. Rotation angles are sampled uniformly from the
given range.

*Scales* are multiplicative factors sampled uniformly from the given
range.

The helper :func:`aug_star_flow` implements the Aug* operator used in
self‑supervised flow training. Given a flow field from frame 1 to
frame 2 and an affine that augments frame 2, it produces the flow
from frame 1 to the augmented frame 2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

# Support absolute import fallback when the package is not available in
# ``sys.path``. When running this module via ``python debugging.py``
# from within the ``optical_flow_research`` directory, Python treats
# the ``augment`` directory as a top‑level package. In that
# situation, the relative import ``from ..ops.flow`` will fail with
# ``ImportError: attempted relative import beyond top‑level package``.
# To make the code robust across all invocation contexts, we attempt
# the relative import first and fall back to a top‑level import. We
# deliberately avoid importing from ``optical_flow_research`` to allow
# running the code directly from the source tree without installing it.
try:
    from ..ops.flow import _base_grid  # type: ignore
except Exception:
    # Fallback when running inside the source tree where ``ops`` is
    # located alongside ``augment``
    from ops.flow import _base_grid  # type: ignore


@dataclass
class TRSRange:
    """Ranges for sampling translation (tx, ty), rotation and scale.

    All values are specified as tuples ``(min, max)``. Translations
    are in pixels; rotations are in degrees; scale is a multiplicative
    factor.
    """

    tx: Tuple[float, float] = (-8.0, 8.0)
    ty: Tuple[float, float] = (-8.0, 8.0)
    rot: Tuple[float, float] = (-5.0, 5.0)
    scale: Tuple[float, float] = (0.98, 1.02)


def _rand_uniform(shape, low, high, device):
    return (high - low) * torch.rand(shape, device=device) + low


def sample_trs(B: int, trs: TRSRange, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample random TRS parameters for a batch of size ``B``.

    Returns four tensors ``(tx, ty, rot, sc)`` of shape ``(B,)`` each.
    """
    tx = _rand_uniform((B,), trs.tx[0], trs.tx[1], device)
    ty = _rand_uniform((B,), trs.ty[0], trs.ty[1], device)
    rot = _rand_uniform((B,), trs.rot[0], trs.rot[1], device)
    sc = _rand_uniform((B,), trs.scale[0], trs.scale[1], device)
    return tx, ty, rot, sc


def build_affine_out2in(tx: torch.Tensor, ty: torch.Tensor, rot_deg: torch.Tensor, scale: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Construct output‑to‑input affine matrices for grid sampling.

    Given batches of translations ``tx, ty`` (in pixels), rotations
    ``rot_deg`` (in degrees) and scales ``scale``, compute the batched
    ``2×3`` matrices that map output coordinates back to input
    coordinates. The transform is centred on the image centre.

    These matrices can be passed directly to
    :func:`torch.nn.functional.affine_grid` with ``align_corners=True``.
    """
    B = tx.shape[0]
    cx = (W - 1.0) / 2.0
    cy = (H - 1.0) / 2.0

    th = rot_deg * math.pi / 180.0
    cos = torch.cos(th)
    sin = torch.sin(th)
    inv_scale = 1.0 / scale

    # Build inverse rotation‑scale matrix (R^T * 1/s)
    a00 =  cos * inv_scale
    a01 =  sin * inv_scale
    a10 = -sin * inv_scale
    a11 =  cos * inv_scale

    # Compute translation such that x_in = A (x_out - c - t) + c
    t0 = cx - (a00 * (cx + tx) + a01 * (cy + ty))
    t1 = cy - (a10 * (cx + tx) + a11 * (cy + ty))

    A = torch.zeros((B, 2, 3), device=tx.device, dtype=torch.float32)
    A[:, 0, 0] = a00
    A[:, 0, 1] = a01
    A[:, 1, 0] = a10
    A[:, 1, 1] = a11
    A[:, 0, 2] = t0
    A[:, 1, 2] = t1
    return A


def invert_affine_2x3(A: torch.Tensor) -> torch.Tensor:
    """Invert a batched pixel‑coordinate affine matrix ``(B,2,3)``."""
    B = A.shape[0]
    M = A[:, :, :2]  # (B,2,2)
    t = A[:, :, 2:]  # (B,2,1)
    Minv = torch.inverse(M)
    tinv = -Minv @ t
    return torch.cat([Minv, tinv], dim=2)


def _to_norm_grid(A_out2in_px: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Convert a pixel affine (output→input) to normalised affine for ``affine_grid``.

    The returned matrix has shape ``(B, 2, 3)`` and can be passed to
    ``torch.nn.functional.affine_grid`` with ``align_corners=True``.
    """
    B = A_out2in_px.shape[0]
    device = A_out2in_px.device
    # Pixel→normalised for input and output
    S_in = torch.tensor([[2.0 / (W - 1.0), 0.0, -1.0], [0.0, 2.0 / (H - 1.0), -1.0]], device=device)
    S_out = torch.tensor([[(W - 1.0) / 2.0, 0.0, (W - 1.0) / 2.0], [0.0, (H - 1.0) / 2.0, (H - 1.0) / 2.0]], device=device)
    # Compose into homogeneous 3×3 matrices
    def to_3x3(A2x3: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((B, 3, 3), device=device, dtype=torch.float32)
        out[:, 2, 2] = 1.0
        out[:, 0:2, 0:3] = A2x3
        return out
    Apx = to_3x3(A_out2in_px)
    Sin = to_3x3(S_in.unsqueeze(0).repeat(B, 1, 1))
    Sout = to_3x3(S_out.unsqueeze(0).repeat(B, 1, 1))
    # x_in_n = Sin * Apx * Sout * x_out_n
    An = Sin @ Apx @ Sout
    return An[:, 0:2, 0:3]


def apply_affine(img: torch.Tensor, A_out2in_px: torch.Tensor) -> torch.Tensor:
    """Apply an affine transform to a batched image tensor ``(B,C,H,W)``.

    Returns an image tensor of the same shape with values sampled using
    bilinear interpolation. Border pixels are replicated. This uses
    ``align_corners=True`` to match the rest of the codebase.
    """
    B, C, H, W = img.shape
    A_norm = _to_norm_grid(A_out2in_px, H, W)
    #print(A_norm)
    grid = F.affine_grid(A_norm, size=img.shape, align_corners=True)
    #print(grid)
    return F.grid_sample(img.cuda(), grid.cuda(), mode="bilinear", padding_mode="zeros", align_corners=True)


def apply_affine_to_coords(coords_b2hw: torch.Tensor, A_in2out_px: torch.Tensor) -> torch.Tensor:
    """Apply an input→output affine to coordinates ``(B,2,H,W)`` in pixel space."""
    B, _, H, W = coords_b2hw.shape
    x = coords_b2hw[:, 0].reshape(B, -1)
    y = coords_b2hw[:, 1].reshape(B, -1)
    ones = torch.ones_like(x)
    pts = torch.stack([x, y, ones], dim=1)  # (B,3,N)
    out = A_in2out_px @ pts  # (B,2,N)
    out = out.reshape(B, 2, H, W)
    return out


def aug_star_flow(flow12: torch.Tensor, A_out2in_px: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the Aug* operator to a flow field.

    Given a flow ``flow12`` from frame 1 to frame 2 and an affine
    transform ``A_out2in_px`` mapping augmented frame 2 back to the
    original frame 2, this function computes the flow from frame 1 to
    the augmented frame 2 under the same augmentation. It also returns a
    mask of pixels for which the transformed coordinates fall inside
    the image bounds. Invalid pixels are marked with zero in the mask.

    Parameters
    ----------
    flow12:
        Flow field from frame 1 to frame 2 with shape ``(B, 2, H, W)``.
    A_out2in_px:
        Affine matrices of shape ``(B, 2, 3)`` mapping augmented
        coordinates to original coordinates (output → input).

    Returns
    -------
    flow_prime:
        Flow field from frame 1 to the augmented frame 2 with shape
        ``(B, 2, H, W)``.
    valid_mask:
        Binary mask of shape ``(B, 1, H, W)`` indicating which pixels
        remain inside the image after transformation.
    """
    B, _, H, W = flow12.shape
    # Invert the output→input affine to get input→output
    A_in2out_px = invert_affine_2x3(A_out2in_px)
    # Base grid of coordinates
    x = _base_grid(B, H, W, flow12.device)
    # y = x + flow(x)
    y = x + flow12
    # Apply affine to target coordinates
    y_prime = apply_affine_to_coords(y, A_in2out_px)
    # Flow from original x to augmented y
    flow_prime = y_prime - x
    # Determine validity (augmented coords inside bounds)
    xp = y_prime[:, 0]
    yp = y_prime[:, 1]
    valid = (xp >= 0.0) & (xp <= (W - 1.0)) & (yp >= 0.0) & (yp <= (H - 1.0))
    return flow_prime, valid.unsqueeze(1).float()
