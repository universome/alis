import sys; sys.path.append('.')
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_utils.ops.fast_bilinear_mult import fast_bilinear_mult


def test_fast_bilinear_mult():
    device = 'cuda'
    batch_size = 5
    h = w = 32
    num_channels = 16

    for dtype in [torch.float16, torch.float32, torch.float64]:
        x = torch.randn(batch_size, num_channels, h, w)
        styles = torch.randn(batch_size, num_channels, 2, 2)

        for device in ['cpu', 'cuda']:
            print(f'Setup: dtype - {dtype}, device - {device}')
            # Slow computation
            x_slow = x.clone().to(device) # [b, c, h, w]
            styles_slow = styles.clone().to(device) # [b, c, 2, 2]
            styles_slow.requires_grad_()
            x_slow.requires_grad_()
            styles_slow_up = F.interpolate(styles_slow, size=x.shape[2:], mode='bilinear', align_corners=True) # [b, c, h, w]
            y_slow = styles_slow_up * x_slow
            loss_slow = y_slow.mean()
            loss_slow.backward()

            # Fast computation
            x_fast = x.clone().to(device) # [b, c, h, w]
            styles_fast = styles.clone().to(device) # [b, c, 2, 2]
            styles_fast.requires_grad_()
            x_fast.requires_grad_()
            y_fast = fast_bilinear_mult(x_fast, styles_fast)
            loss_fast = y_fast.mean()
            loss_fast.backward()

            # Checking the computation
            assert torch.allclose(y_slow, y_fast, atol=1e-6)
            assert torch.allclose(x_slow.grad, x_fast.grad)
            assert torch.allclose(styles_slow.grad, styles_fast.grad)
