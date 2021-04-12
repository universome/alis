import sys; sys.path.append('.')
from time import time

import torch
import numpy as np

from training.layers import generate_coords
from training.layers import generate_logarithmic_basis


def test_coords_generation():
    batch_size = 3
    img_size = 32
    coords = generate_coords(batch_size, img_size)
    img_coords = coords[0].permute(1, 0).view(img_size, img_size, 2)
    coords_range = (-1.0, 1.0) # Right bound is __not__ inclusive
    coords_step = (coords_range[1] - coords_range[0]) / img_size
    max_coord_value = coords_range[0] + coords_step * (img_size - 1)

    assert coords.shape == (batch_size, 2, img_size * img_size)
    assert torch.equal(
        img_coords[0][0], # Left upper corner
        torch.tensor([-1.0, max_coord_value]))
    assert torch.allclose(
        img_coords[0][(img_size - 1)], # Right upper corner
        torch.tensor([max_coord_value, max_coord_value]))
    assert torch.allclose(
        img_coords[(img_size - 1)][0], # Left lower corner
        torch.tensor([-1.0, -1.0]))
    assert torch.allclose(
        img_coords[(img_size - 1)][(img_size - 1)], # Right lower corner
        torch.tensor([max_coord_value, -1.0]))


def test_coords_mm_einsum():
    batch_size = 4
    emb_dim = 8
    coord_dim = 2
    img_size = 16
    W = torch.randn(batch_size, emb_dim, coord_dim)
    coords = torch.randn(batch_size, coord_dim, img_size, img_size)

    res_normal = (W @ coords.view(batch_size, coord_dim, -1))
    res_einsum = torch.einsum('bdc,bcxy->bdxy', W, coords)

    torch.allclose(res_normal, res_einsum.view(batch_size, emb_dim, -1))


def test_log_basis():
    img_size = 64
    coords = generate_coords(1, img_size)
    basis = generate_logarithmic_basis(img_size)
    coords = coords.view(2, img_size ** 2)
    feats = (basis @ coords).sin()
    feats = feats.view(basis.shape[0], img_size, img_size)

    for f in feats:
        assert f.max() == 1.0
        assert f.min() == -1.0
