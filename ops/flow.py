"""
Flow operators
==============

This module defines a few basic operations on optical flow fields
represented as PyTorch tensors.  The functions here are used by
training, evaluation and augmentation code to compose flows, warp one
flow field by another and compute validity masks.  All flows are
assumed to be in pixel coordinates with shape ``(B,2,H,W)`` where the
first channel stores the displacement in the x direction (columns)
and the second channel stores the displacement in the y direction
(rows).

The warping and composition functions rely on ``torch.nn.functional.grid_sample``
for differentiable sampling of tensors.  Bilinear interpolation is
employed with border replication for out‑of‑bounds sampling.  A
consistency mask can be derived by checking whether the target
coordinates after adding the flow remain inside the image domain.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _base_grid(B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
    """Create a base coordinate grid.

    Returns a tensor of shape ``(B,2,H,W)`` where ``grid[:,0]``
    contains the x coordinates (column indices) and ``grid[:,1]``
    contains the y coordinates (row indices).  The grid is replicated
    across the batch dimension and lives on the specified device.
    """
    # Note: ``torch.meshgrid`` returns coordinate matrices such that
    # ``yy[i,j] = ys[i]`` and ``xx[i,j] = xs[j]`` when using the
    # default indexing.  This yields a grid where the first output
    # corresponds to the y (row) coordinate and the second to x
    # (column).  We stack them in (x,y) order to match the flow
    # convention.
    ys = torch.arange(H, device=device)
    xs = torch.arange(W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # shape (H,W)
    # Stack into (2,H,W) and expand batch dimension
    grid = torch.stack([xx.float(), yy.float()], dim=0)  # (2,H,W)
    grid = grid.unsqueeze(0).expand(B, 2, H, W).clone()
    return grid


def flow_warp(flow: torch.Tensor, flow_ref: torch.Tensor) -> torch.Tensor:
    """Warp ``flow`` by ``flow_ref``.

    Given a flow field ``flow`` representing displacements from an
    intermediate image to a reference image, and another flow field
    ``flow_ref`` representing displacements from the input image to
    that intermediate image, this function returns a new flow field
    defined on the input image.  The warping is differentiable and
    uses bilinear interpolation with border padding.

    Parameters
    ----------
    flow: torch.Tensor
        Flow field of shape ``(B,2,H,W)`` to be sampled.
    flow_ref: torch.Tensor
        Flow field of shape ``(B,2,H,W)`` providing target coordinates.

    Returns
    -------
    torch.Tensor
        Warped flow of shape ``(B,2,H,W)``.
    """
    B, C, H, W = flow.shape
    # Compute target sampling coordinates: base grid + reference flow
    device = flow.device
    coords = _base_grid(B, H, W, device) + flow_ref  # (B,2,H,W)
    # Normalise coordinates to [-1,1] for grid_sample
    x = coords[:, 0]  # (B,H,W)
    y = coords[:, 1]  # (B,H,W)
    # Avoid division by zero when W or H equals 1
    if W > 1:
        x_norm = 2.0 * x / float(W - 1) - 1.0
    else:
        x_norm = torch.zeros_like(x)
    if H > 1:
        y_norm = 2.0 * y / float(H - 1) - 1.0
    else:
        y_norm = torch.zeros_like(y)
    grid = torch.stack((x_norm, y_norm), dim=-1)  # (B,H,W,2)
    # Sample the flow using bilinear interpolation; replicate border for out‑of‑bounds
    warped = F.grid_sample(flow, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return warped


def flow_compose(flow12: torch.Tensor, flow23: torch.Tensor) -> torch.Tensor:
    """Compose two flow fields.

    Given ``flow12`` (displacements from image 1 to image 2) and
    ``flow23`` (displacements from image 2 to image 3), return the
    composed flow ``flow13_hat`` which maps directly from image 1 to
    image 3.  The composition is computed as::

        flow13_hat = flow12 + flow_warp(flow23, flow12)

    This corresponds to first following ``flow12`` to reach image 2 and
    then following ``flow23`` from there.
    """
    return flow12 + flow_warp(flow23, flow12)


def in_bounds_mask(flow: torch.Tensor) -> torch.Tensor:
    """Compute a validity mask for flow sampling.

    For each pixel in the input image, this mask indicates whether
    adding the flow would result in a coordinate that lies inside the
    image bounds.  The returned mask has shape ``(B,1,H,W)`` with
    values 1.0 for valid pixels and 0.0 for invalid ones.  This mask
    does *not* account for occlusions or other scene‑specific
    phenomena; it purely checks geometric bounds.
    """
    B, _, H, W = flow.shape
    device = flow.device
    coords = _base_grid(B, H, W, device) + flow
    x = coords[:, 0]  # (B,H,W)
    y = coords[:, 1]
    valid = (x >= 0.0) & (x <= float(W - 1)) & (y >= 0.0) & (y <= float(H - 1))
    return valid.unsqueeze(1).float()


__all__ = ["_base_grid", "flow_warp", "flow_compose", "in_bounds_mask"]