from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F

from torch_utils import misc

#----------------------------------------------------------------------------

def fast_bilinear_mult(x, styles):
    """
    x: [b, c, h, w],
    styles: [b, c, 2, 2]
    """
    b, c, h, w = x.shape
    misc.assert_shape(styles, [b, c, 2, 2])

    kwargs = dict(device=x.device, dtype=x.dtype)
    top_to_bottom = torch.linspace(1, 0, h, **kwargs).unsqueeze(1)
    left_to_right = torch.linspace(1, 0, w, **kwargs).unsqueeze(0)
    coefs_11 = top_to_bottom * left_to_right # [h, w]
    coefs_12 = top_to_bottom * (1.0 - left_to_right) # [h, w]
    coefs_21 = (1.0 - top_to_bottom) * left_to_right  # [h, w]
    coefs_22 = (1.0 - top_to_bottom) * (1.0 - left_to_right) # [h, w]
    coefs = torch.stack([coefs_11, coefs_12, coefs_21, coefs_22]) # [4, h, w]
    coefs = coefs.unsqueeze(0).unsqueeze(2) # [1, 4, 1, h, w]
    xs = (x.unsqueeze(1) * coefs) # [b, 4, c, h, w]
    styles = styles.permute(0, 2, 3, 1).view(b, 4, c) # [b, 4, c]
    styles = styles.view(b, 4, c, 1, 1) # [b, 4, c, 1, 1]
    y = (xs * styles).sum(dim=1) # [b, c, h, w]

    return y


def fast_bilinear_mult_row(x: Tensor, styles: Tensor, shifts: Optional[Tensor]=None) -> Tensor:
    b, c, h, w = x.shape
    context_size = 2
    misc.assert_shape(styles, [b, c, context_size + 1])

    centers = shifts
    if centers is None:
        centers = torch.zeros(b, 2, dtype=styles.dtype, device=styles.device)

    misc.assert_shape(centers, [b, 2])
    assert centers.min().item() >= -1.0
    assert centers.max().item() >= -1.0

    # Centers are [-1, 1] range, but w_before/w_after positions correspond to -2/2.
    # Constructing the bounds for each center
    # The size of the square is 2: it is in [-1, 1] x [-1, 1]
    # Bounds correspond to left and right borders
    assert context_size == 2
    bounds = torch.stack([
        torch.stack([centers[:, 0] - 1, centers[:, 1]], dim=1),
        torch.stack([centers[:, 0] + 1, centers[:, 1]], dim=1)
    ], dim=1) # [b, 2, 2]
    bounds = bounds.unsqueeze(1) # [b, 1, 2, 2] == [b, h, w, 2]

    # Now, grid sample assume [-1, 1] range, so adjust:
    bounds.mul_(0.5)

    # Also, for F.grid_sample we need to flip y coordinate
    bounds[:, :, :, 1].mul_(-1.0)

    # Now, we can get our interpolated embeddings
    w_bounds = F.grid_sample(styles.unsqueeze(2), bounds.to(styles.dtype), mode='bilinear', align_corners=True) # [b, c, 1, 2]

    # Now, we can interpolate and modulate
    modulation = F.interpolate(w_bounds, size=(1, w), mode='bilinear', align_corners=True) # [b, c, 1, w]
    x = x * modulation # [b, c, h, w]

    # print('PERFORMED fast_bilinear_mult_row')

    return x


def fast_manual_bilinear_mult_row(x: Tensor, styles: Tensor, left_borders_idx: Tensor, grid_size: int, w_coord_dist: float, w_lerp_multiplier: float=1.0) -> Tensor:
    b, c, h, w = x.shape
    misc.assert_shape(styles, [b, 3, c])
    misc.assert_shape(left_borders_idx, [b])

    w_dist = int(0.5 * w_coord_dist * w)
    interp_coefs = torch.linspace(1 / (2 * w_dist), 1 - 1 / (2 * w_dist), w_dist, device=x.device, dtype=styles.dtype) # [w_dist]
    interp_coefs = interp_coefs * w_lerp_multiplier
    interp_coefs = interp_coefs.view(1, w_dist, 1) # [1, w_dist, 1]
    styles_grid_left = styles[:, 0].unsqueeze(1) * (w_lerp_multiplier - interp_coefs) + styles[:, 1].unsqueeze(1) * interp_coefs # [b, w_dist, c]
    styles_grid_right = styles[:, 1].unsqueeze(1) * (w_lerp_multiplier - interp_coefs) + styles[:, 2].unsqueeze(1) * interp_coefs # [b, w_dist, c]
    styles_grid = torch.cat([styles_grid_left, styles_grid_right], dim=1).to(x.dtype) # [b, 2 * w_dist, c]

    # Left borders were randomly sampled in [0, 2 * w_dist - w] integer range
    # We use them to select the corresponding styles
    patch_size = w // grid_size
    batch_idx = torch.arange(b, device=x.device).view(-1, 1).repeat(1, w) # [b, w]
    grid_idx = (left_borders_idx.unsqueeze(1) * patch_size) + torch.arange(w, device=x.device).view(1, -1) # [b, w]
    latents = styles_grid[batch_idx, grid_idx].permute(0, 2, 1) # [b, c, w]
    x = x * latents.unsqueeze(2) # [b, c, h, w]

    return x
