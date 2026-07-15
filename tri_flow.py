from augment.trs import *
from augment import _base_grid
from ops import flow_compose, in_bounds_mask


"""
This file contains the main functions to be imported to apply
the 3 consistency losses.

To implement the 3 consistency losses, 
import the
'get_aug_img_flow' for augmentation consistency,
'loss_identity' for the cycle consistency loss, and
'loss_composition' for the temporal compositional consistency.


'get_aug_img_flow'
function takes in the image to augment, the flows, device and command line arguments. 
The function returns:
the augmented image,
the analytically augmented flow,
validity mask.

Use your model to predict the flow from img1 to the aug_img2, then calculate
the difference between these predictions optionally using the validity mask.

'loss_identity' function.
Calculate the flow from img1 -> img2, and from img2 -> img1.
Then use this function to calculate the identity loss.
This function returns a scalar valued tensor.

'loss_composition' function.
To use this loss, your your data class needs to be able to output 3 frames.
Calculate the following flows:
f13, f31, f12, f23.
This function will return the composed flow f13_hat, loss, and validity mask.
"""


def epe_map(flow_diff: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute the per‑pixel endpoint error (EPE).

    Given a difference of flows ``flow_diff`` with shape ``(B, 2, H, W)``
    this returns a tensor of shape ``(B, 1, H, W)`` where each
    element is ``sqrt(dx^2 + dy^2 + eps)``. A small epsilon is added to
    ensure differentiability at zero.
    """
    return torch.sqrt((flow_diff * flow_diff).sum(dim=1, keepdim=True) + eps)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute the mean of ``x`` over ``mask``.

    ``x`` may have shape ``(B,H,W)`` or ``(B,1,H,W)`` and ``mask`` has
    shape ``(B,1,H,W)``. The returned tensor is scalar valued. If the
    mask is empty the denominator is clamped to ``eps`` to avoid
    division by zero.
    """
    if x.dim() == 3:
        x = x.unsqueeze(1)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    num = (x * mask).sum()
    den = mask.sum().clamp_min(eps)
    return num / den


def transform_flow_with_matrix(f01: torch.Tensor, A_out2in: torch.Tensor) -> torch.Tensor:
    """
    Transforms optical flow f01 (from I0 to I1) to match the affine-transformed I1.

    Inputs:
        f01: Tensor of shape (B, 2, H, W) : flow from I0 to I1
        A_out2in: Tensor of shape (B, 2, 3) : affine matrices from augmented I1 back to original I1

    Returns:
        f01_prime: flow from I0 to augmented I1
        valid_mask: mask to invalidate pixels landing outside image bounds
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

# Augmentation Consistency
def get_aug_img_flow(img2, flow, device, args):
    """
    Get the augmented image and analytically computed augmented flow.

    Returns:
        img2_aug: (B, C, H, W)
        flow_aug: (B, 2, H, W)
        valid_aug: (B, 1, H, W)
        
    """
    
    try: 
        trs = TRSRange(
                tx=(-args.aug_tx, args.aug_tx),
                ty=(-args.aug_ty, args.aug_ty),
                rot=(-args.aug_rot, args.aug_rot),
                scale=(args.aug_smin, args.aug_smax),
            )
    except AttributeError:
        trs = TRSRange(
                tx=(-2, 2),
                ty=(-2, 2),
                rot=(-4, 4),
                scale=(0.95, 1.05),
            )
        

    B, _, H, W = img2.shape

    trs_params = sample_trs(B, trs, device)

    A_out2in = build_affine_out2in(*trs_params, H, W)

    img2_aug = apply_affine(img2, A_out2in)

    flow_aug, valid_aug = transform_flow_with_matrix(flow, A_out2in)

    return img2_aug, flow_aug, valid_aug


# Cycle Consistency
def loss_identity(flow12: torch.Tensor, flow21: torch.Tensor, threshold) -> torch.Tensor:
    """
    Cycle consistency (round-trip) loss:
        flow_12(x) + flow_21(x + flow_12(x)) ~= 0

    This loss encourages the round-trip flow ``flow12 ∘ flow21`` to be
    zero: applying ``flow21`` after ``flow12`` should map points back
    to their origin. Pixels that would flow out of bounds after
    ``flow12`` are masked out.

    Returns:
        loss: torch.Tensor
    """

    resid = epe_map(flow_compose(flow12, flow21))
    mask = in_bounds_mask(flow12)
    mask[resid>threshold] = 0
    return masked_mean(resid, mask)


# Temporal Compositional Consistency
def loss_composition(flow12: torch.Tensor, flow23: torch.Tensor, flow13: torch.Tensor, flow31: torch.Tensor, threshold) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Triangle / composition consistency loss.

    Compose flow_ab with flow_bc:
        v_ac(x) = v_ab(x) + v_bc(x + v_ab(x))

    Returns:
        flow13_hat: (B,2,H,W)
        mean_epe: (1)
        mask: (B, 1, H, W)
    """

    flow13_hat = flow_compose(flow12, flow23)

    resid = epe_map(flow_compose(flow13, flow31))


    diff = epe_map(flow13_hat - flow13)
    mask = in_bounds_mask(flow12)
    mask[diff>threshold] = 0
    mask[resid>threshold] = 0

    return flow13_hat, masked_mean(diff, mask), mask