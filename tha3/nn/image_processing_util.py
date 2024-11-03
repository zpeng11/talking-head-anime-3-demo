import torch
from torch import Tensor
from torch.nn.functional import affine_grid, grid_sample
import torch.nn.functional as F

# Affine_grid patch support from https://github.com/pytorch/pytorch/issues/30563#issuecomment-1295761759
def affine_grid(theta, size, align_corners=False):
    device = theta.device
    dtype = theta.dtype
    N, C, H, W = size
    grid = create_grid(N, C, H, W, device, dtype)
    grid = grid.view(N, H * W, 3).bmm(theta.transpose(1, 2))
    grid = grid.view(N, H, W, 2)
    return grid

def create_grid(N, C, H, W, device, dtype):
    grid = torch.empty((N, H, W, 3), device= device, dtype=dtype)
    grid.select(-1, 0).copy_(linspace_from_neg_one(W, device=device, dtype=dtype))
    grid.select(-1, 1).copy_(linspace_from_neg_one(H, device=device, dtype=dtype).unsqueeze_(-1))
    grid.select(-1, 2).fill_(1)
    return grid
    
def linspace_from_neg_one(num_steps, device, dtype):
    r = torch.linspace(-1, 1, num_steps, dtype=dtype, device=device)
    r = r * (num_steps - 1) / num_steps
    return r

def patch_affine_grid_generator():
    torch.nn.functional.affine_grid = affine_grid
patch_affine_grid_generator()

def apply_rgb_change(alpha: Tensor, color_change: Tensor, image: Tensor):
    image_rgb = image[:, 0:3, :, :]
    color_change_rgb = color_change[:, 0:3, :, :]
    output_rgb = color_change_rgb * alpha + image_rgb * (1 - alpha)
    return torch.cat([output_rgb, image[:, 3:4, :, :]], dim=1)


def apply_grid_change(grid_change, image: Tensor) -> Tensor:
    n, c, h, w = image.shape
    device = grid_change.device
    grid_change = torch.transpose(grid_change.view(n, 2, h * w), 1, 2).view(n, h, w, 2)
    identity = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=grid_change.dtype,
        device=device).unsqueeze(0).repeat(n, 1, 1)
    base_grid = affine_grid(identity, [n, c, h, w], align_corners=False)
    grid = base_grid + grid_change
    resampled_image = grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=False)
    return resampled_image


class GridChangeApplier:
    def __init__(self):
        self.last_n = None
        self.last_device = None
        self.last_identity = None

    def apply(self, grid_change: Tensor, image: Tensor, align_corners: bool = False) -> Tensor:
        n, c, h, w = image.shape
        device = grid_change.device
        grid_change = torch.transpose(grid_change.view(n, 2, h * w), 1, 2).view(n, h, w, 2)

        if n == self.last_n and device == self.last_device:
            identity = self.last_identity
        else:
            identity = torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=grid_change.dtype,
                device=device,
                requires_grad=False) \
                .unsqueeze(0).repeat(n, 1, 1)
            self.last_identity = identity
            self.last_n = n
            self.last_device = device
        base_grid = affine_grid(identity, [n, c, h, w], align_corners=align_corners)

        grid = base_grid + grid_change
        resampled_image = grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=align_corners)
        return resampled_image


def apply_color_change(alpha, color_change, image: Tensor) -> Tensor:
    return color_change * alpha + image * (1 - alpha)
