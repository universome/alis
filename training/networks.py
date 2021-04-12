# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from typing import Tuple, Optional, Callable, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from omegaconf import OmegaConf, DictConfig

from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma
from torch_utils.ops.fast_bilinear_mult import fast_manual_bilinear_mult_row

from .layers import (
    FullyConnectedLayer,
    GenInput,
    ModulatedCoordFuser,
    fmm_modulate,
)


#----------------------------------------------------------------------------

@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

#----------------------------------------------------------------------------

@misc.profiled_function
def modulated_conv2d(
    x,                              # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,                         # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,                         # Modulation coefficients of shape [batch_size, in_channels].
    noise               = None,     # Optional noise tensor to add to the output activations.
    up                  = 1,        # Integer upsampling factor.
    down                = 1,        # Integer downsampling factor.
    padding             = 0,        # Padding with respect to the upsampled image.
    resample_filter     = None,     # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate          = True,     # Apply weight demodulation?
    flip_weight         = True,     # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv       = True,     # Perform modulation, convolution, and demodulation as a single fused operation?
    fmm_weights         = None,     # If FMM weights provided, then applies them _after_ the styles modulation
    fmm_mod_type        = None,     # FMM modulation type: ["mult", "add"]
    fmm_add_weight      = 1.0,      # You can specify the weight of the additive FMM term
    fmm_activation      = None,     # Wheather or not we should use the activation
    upsampling_mode     = None,     # When upsampled with bilinear/nearest, we do this outside conv2d_resample
    spatial_style       = False,    # Should we use spatial modulation or the traditional one?
    left_borders_idx    = None,     # For spatial style, we use left_borders_idx to compute shifts
    grid_size           = None,     # Grid size for patch wise op
    w_lerp_kwargs       = {},       # Arguments for fast_manual_bilinear_mult_row
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw]) # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]

    if spatial_style:
        context_size = 2
        misc.assert_shape(styles, [batch_size, context_size + 1, in_channels])
    else:
        misc.assert_shape(styles, [batch_size, in_channels]) # [NI]

    # If custom upsampling is enabled, then we do it separately
    up_for_conv2d = up if upsampling_mode is None else 1

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
        dims = [2] if spatial_style else [1]
        styles = styles / styles.norm(float('inf'), dim=dims, keepdim=True) # max_I

    # Execute as one fused op using grouped convolution.
    w, demod_coefs = prepare_weights(
        weight,
        styles,
        demodulate,
        fused_modconv,
        batch_size,
        fmm_weights,
        fmm_mod_type,
        fmm_add_weight,
        fmm_activation,
        spatial_style,
    )

    if fused_modconv:
        with misc.suppress_tracer_warnings(): # this value will be treated as a constant
            batch_size = int(batch_size)

        misc.assert_shape(x, [batch_size, in_channels, None, None])
        x = x.reshape(1, -1, *x.shape[2:])
        w = w.reshape(-1, in_channels, kh, kw)
        x = conv2d_resample.conv2d_resample(
            x=x,
            w=w.to(x.dtype),
            f=resample_filter,
            up=up_for_conv2d,
            down=down,
            padding=padding,
            groups=batch_size,
            flip_weight=flip_weight)
        x = x.reshape(batch_size, -1, *x.shape[2:])
        x = maybe_upsample(x, upsampling_mode, up)

        if noise is not None:
            x = x.add_(noise)
    else:
        # Execute by scaling the activations before and after the convolution.
        if spatial_style:
            x = fast_manual_bilinear_mult_row(x, styles, left_borders_idx, **w_lerp_kwargs)
        else:
            x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)

        x = conv2d_resample.conv2d_resample(
            x=x,
            w=weight.to(x.dtype),
            f=resample_filter,
            up=up_for_conv2d,
            down=down,
            padding=padding,
            flip_weight=flip_weight)
        x = maybe_upsample(x, upsampling_mode, up)

        if demodulate and noise is not None:
            x = fma.fma(x, demod_coefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * demod_coefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))

    return x


@misc.profiled_function
def patchwise_conv2d(
    x,                              # Input tensor of shape [batch_size, in_channels, h, w].
    weight,                         # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,                         # Modulation coefficients of shape [batch_size, 3, in_channels].
    noise               = None,     # Optional noise tensor to add to the output activations.
    up                  = 1,        # Integer upsampling factor.
    padding             = 0,        # Padding with respect to the upsampled image.
    resample_filter     = None,     # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate          = True,     # Apply weight demodulation?
    flip_weight         = True,     # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    left_borders_idx    = None,     # Left borders for spatial w
    grid_size: int      = None,     # Size of the grid
    mod_end_idx: int    = None,     # Modulate only x[:, :mod_end_idx] part of x
    instance_norm: bool = False,     # Should we apply InstanceNorm to x?
    w_lerp_kwargs       = {},       # Arguments for fast_manual_bilinear_mult_row
):
    b, c_in, h, w = x.shape
    c_out, c_in, kh, kw = weight.shape
    misc.assert_shape(weight, [c_out, c_in, kh, kw]) # [OIkk]
    misc.assert_shape(x, [b, c_in, None, None]) # [NIHW]
    misc.assert_shape(styles, [b, 3, c_in])
    # misc.assert_shape(styles, [b, 3, mod_end_idx])

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(c_in * kh * kw) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=[2], keepdim=True) # max_I

    if instance_norm and h > grid_size * 2:
        x_mod, x_coord = x[:, :mod_end_idx], x[:, mod_end_idx:]
        x_mod = patchwise_op(lambda x: x / (x.std(dim=[2, 3], keepdim=True) + 1e-8), x_mod, grid_size)
        x = torch.cat([x_mod, x_coord], dim=1)

    demod_coefs = (weight.square().sum(dim=[1,2,3]) + 1e-8).rsqrt() # [O]
    demod_coefs = demod_coefs.unsqueeze(0).repeat(b, 1) # [NO]

    assert mod_end_idx is not None
    x = fast_manual_bilinear_mult_row(x, styles.to(x.dtype), left_borders_idx, grid_size, **w_lerp_kwargs) # [b, c_in, h, w]
    # x_mod, x_coord = x[:, :mod_end_idx], x[:, mod_end_idx:]
    # x_mod = fast_manual_bilinear_mult_row(x_mod, styles.to(x.dtype), left_borders_idx, grid_size, **w_lerp_kwargs) # [b, c_in, h, w]
    # x = torch.cat([x_mod, x_coord], dim=1)

    # Convert x from [b, c, h, w] into [b * grid_size, c_in, h, w_patch]

    x = patchwise_op(
        conv2d_resample.conv2d_resample,
        x,
        grid_size,
        __up=up,
        w=weight.to(x.dtype),
        f=resample_filter,
        up=up,
        padding=padding,
        flip_weight=flip_weight)

    if demodulate and noise is not None:
        x = fma.fma(x, demod_coefs.to(x.dtype).reshape(b, -1, 1, 1), noise.to(x.dtype))
    elif demodulate:
        x = x * demod_coefs.to(x.dtype).reshape(b, -1, 1, 1)
    elif noise is not None:
        x = x.add_(noise.to(x.dtype))

    return x


@misc.profiled_function
def patchwise_op(op: Callable, x: Tensor, grid_size: int, __up: int=1, *args, **kwargs) -> Any:
    # Convert x from [b, c, h, w] into [b * grid_size, c_in, h, w_patch]
    b, c_in, h, w = x.shape
    w_patch = w // grid_size
    x = x.view(b, c_in, h, grid_size, w_patch) # [b, c_in, h, grid_size, w_patch]
    x = x.permute(0, 3, 1, 2, 4) # [b, grid_size, c_in, h, w_patch]
    x = x.contiguous().view(b * grid_size, c_in, h, w_patch) # [b * grid_size, c_in, h, w_patch]

    # Applying the operation
    y = op(x, *args, **kwargs)

    # Convert y back from [b * grid_size, c_out, h_patch * grid_size, w_patch] to [b, c_out, h, w]
    c_out = y.shape[1]
    y = y.view(b, grid_size, c_out, h * __up, w_patch * __up) # [b, grid_size, c_out, h * __up, w_patch * __up]
    y = y.permute(0, 2, 3, 1, 4) # [b, c_out, h * __up, grid_size, w_patch * __up]
    y = y.contiguous().view(b, c_out, h * __up, w * __up) # [b, c_out, h, grid_size, w_patch]

    return y


@misc.profiled_function
def prepare_weights(
        weight: Tensor,
        styles: Tensor,
        demodulate: bool,
        fused_modconv: bool,
        batch_size: int,
        fmm_weights: Optional[Tensor],
        fmm_mod_type: Optional[str],
        fmm_add_weight: Optional[Tensor],
        fmm_activation: Optional[str],
        spatial_style: bool) -> Tuple[Tensor, Optional[Tensor]]:

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    demod_coefs = None

    if demodulate or fused_modconv:
        w = weight.unsqueeze(0) # [NOIkk]

        if spatial_style:
            assert fused_modconv is False, "Cannot apply spatial styles while fusing"
            w = w.repeat(batch_size, 1, 1, 1, 1) # [NOIkk]
        else:
            w = w * styles.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]

    if not fmm_weights is None:
        assert fused_modconv
        # Demodulating after FMM only if we won't demodulate later here
        w = fmm_modulate(
            w,
            fmm_weights,
            fmm_mod_type,
            demodulate=not demodulate,
            fmm_add_weight=fmm_add_weight,
            activation=fmm_activation)

    if demodulate:
        demod_coefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]

    if demodulate and fused_modconv:
        w = w * demod_coefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]

    return w, demod_coefs


@misc.profiled_function
def maybe_upsample(x, upsampling_mode: str=None, up: int=1) -> Tensor:
    if upsampling_mode in ['nearest', 'bilinear'] and up > 1:
        x = F.interpolate(x, mode=upsampling_mode, align_corners=True, scale_factor=up)

    return x


@misc.profiled_function
def run_spatial_affine(affine_layer: FullyConnectedLayer, w: Tensor, w_context: Tensor, fallback_to_mean: bool=False) -> Tensor:
    context_size, batch_size, w_dim = 2, w.shape[0], w.shape[1]
    misc.assert_shape(w_context, [batch_size, context_size, w_dim])
    w_grid = torch.stack([w_context[:, 0], w, w_context[:, 1]], dim=1) # [batch_size, 3, w_dim]
    w_grid = w_grid.view(batch_size * (context_size + 1), w_dim)
    styles = affine_layer(w_grid) # [b * (context_size + 1), c]
    styles = styles.reshape(batch_size, context_size + 1, -1) # [b, context_size + 1, c]
    styles = styles.contiguous()

    if fallback_to_mean:
        styles = styles.mean(dim=[1], keepdim=True).repeat(1, context_size + 1, 1)
    else:
        pass # We will perform fast_bilinear_mult later

    return styles

#----------------------------------------------------------------------------

@persistence.persistent_class
class Conv2dLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        kernel_size,                    # Width and height of the convolution kernel.
        bias            = True,         # Apply additive bias before the activation function?
        activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
        up              = 1,            # Integer upsampling factor.
        down            = 1,            # Integer downsampling factor.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
        channels_last   = False,        # Expect the input to have memory_format=channels_last?
        trainable       = True,         # Update the weights of this layer during training?
    ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1) # slightly faster
        x = conv2d_resample.conv2d_resample(
            x=x,
            w=w.to(x.dtype),
            f=self.resample_filter,
            up=self.up,
            down=self.down,
            padding=self.padding,
            flip_weight=flip_weight)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)

        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        h_dim           = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.995,    # Decay for tracking the moving average of W during training, None = do not track.
        num_modes       = 1,        # Should we use a multi-modal training?
        mode_max_norm   = 1,        # Max norm for the mode embeddings
    ):
        super().__init__()

        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta
        self.num_modes = num_modes

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if h_dim is None:
            h_dim = w_dim
        features_list = [z_dim + embed_features] + [h_dim] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)

        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

        if num_modes > 1:
            self.modes_embs = torch.nn.Embedding(num_modes, z_dim, max_norm=mode_max_norm)
            # If we wont do this, then our `check_ddp_consistency` will fail
            # self.modes_embs.weight.data = 0.99 * max_norm * F.normalize(self.modes_embs.weight.data, dim=1, p=2)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False, modes_idx=None):
        if self.num_modes > 1:
            assert modes_idx is not None
            z = (z + self.modes_embs(modes_idx)) / np.sqrt(2)
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x


#----------------------------------------------------------------------------

@persistence.persistent_class
class PatchWiseSynthesisLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        w_dim,                          # Intermediate latent (W) dimensionality.
        resolution,                     # Resolution of this layer.
        kernel_size     = 3,            # Convolution kernel size.
        up              = 1,            # Integer upsampling factor.
        use_noise       = True,         # Enable noise input?
        activation      = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        channels_last   = False,        # Use channels_last format for the weights?
        coord_dim_size  = 0,
        cfg             = {},           # Synthesis config
    ):
        super().__init__()

        self.cfg = OmegaConf.create(cfg)
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = (0 if self.cfg.force_zero_padding else kernel_size // 2)
        self.act_gain = bias_act.activation_funcs[activation].def_gain
        self.in_channels = in_channels
        self.coord_dim_size = coord_dim_size

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        # self.num_mod_channels = in_channels - coord_dim_size # Do not modulate coords
        # self.affine = FullyConnectedLayer(w_dim, self.num_mod_channels, bias_init=1)

        if use_noise:
            self.register_buffer('noise_const', torch.randn([self.resolution, self.resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))

        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, noise_mode='random', gain=1, w_context: Optional[Tensor]=None, left_borders_idx: Optional[Tensor]=None, **legacy_kwargs):
        assert noise_mode in ['random', 'const', 'none']
        misc.assert_shape(x, [None, self.weight.shape[1], self.resolution // self.up, self.resolution // self.up])

        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        styles = run_spatial_affine(self.affine, w, w_context, fallback_to_mean=self.cfg.style.fallback_to_mean)
        flip_weight = (self.up == 1) # slightly faster

        x = patchwise_conv2d(
            x=x,
            weight=self.weight,
            styles=styles,
            noise=noise,
            up=self.up,
            padding=self.padding,
            resample_filter=self.resample_filter,
            flip_weight=flip_weight,
            left_borders_idx=left_borders_idx,
            grid_size=self.cfg.patchwise.grid_size,
            instance_norm=self.cfg.patchwise.instance_norm,
            mod_end_idx=(x.shape[1] - self.coord_dim_size),

            w_lerp_kwargs=dict(
                w_coord_dist=self.cfg.patchwise.w_coord_dist,
                w_lerp_multiplier=self.cfg.patchwise.w_lerp_multiplier,
            ),
        )

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)

        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        w_dim,                          # Intermediate latent (W) dimensionality.
        resolution,                     # Resolution of this layer.
        kernel_size     = 3,            # Convolution kernel size.
        up              = 1,            # Integer upsampling factor.
        use_noise       = True,         # Enable noise input?
        activation      = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        channels_last   = False,        # Use channels_last format for the weights?
        down: int       = 1,            # Wheather or not we should downsample. We cannot have up > 1 and down > 1 at the same time
        cfg             = {},           # Synthezis config
    ):
        super().__init__()

        self.cfg = OmegaConf.create(cfg)
        self.resolution = resolution
        self.up = up
        self.down = down
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = (0 if self.cfg.force_zero_padding else kernel_size // 2)
        self.act_gain = bias_act.activation_funcs[activation].def_gain
        self.in_channels = in_channels

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))

        if self.cfg.fmm.enabled and self.cfg.fmm.ranks.get(str(self.resolution), 0) > 0:
            rank = fmm_cfg['ranks'][str(self.resolution)]
            self.fmm_projection = FullyConnectedLayer(w_dim, (in_channels + out_channels) * rank, bias_init=0)
            self.fmm_mod_type = fmm_cfg['mod_type']

            if fmm_cfg.get('use_rezero_fmm_add', False) and fmm_cfg['mod_type'] == 'add':
                self.fmm_add_weight = torch.nn.Parameter(torch.tensor([0.0]))
            else:
                self.fmm_add_weight = 1.0

            self.fmm_activation = fmm_cfg.get('activation', 'linear')
        else:
            self.fmm_projection = None
            self.fmm_mod_type = None
            self.fmm_add_weight = None
            self.fmm_activation = None

            # Fallback to the classical modulation
            self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)

        if use_noise:
            self.register_buffer('noise_const', torch.randn([self.resolution, self.resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))

        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1, w_context: Optional[Tensor]=None, left_borders_idx=None):
        assert noise_mode in ['random', 'const', 'none']

        misc.assert_shape(x, [None, self.weight.shape[1], self.resolution // self.up, self.resolution // self.up])

        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        if self.fmm_projection is None:
            if self.cfg.style.is_spatial:
                styles = run_spatial_affine(self.affine, w, w_context, fallback_to_mean=self.cfg.style.fallback_to_mean)

                if not self.cfg.style.fallback_to_mean:
                    fused_modconv = False
            else:
                styles = self.affine(w)

            fmm_weights = None
        else:
            styles = torch.ones(x.shape[0], self.in_channels, device=x.device)
            fmm_weights = self.fmm_projection(w)
            fused_modconv = True

            assert not self.cfg.style.is_spatial

        flip_weight = (self.up == 1) # slightly faster

        if self.cfg.style.is_spatial:
            fused_modconv = False

        x = modulated_conv2d(
            x=x,
            weight=self.weight,
            styles=styles,
            noise=noise,
            up=self.up,
            down=self.down,
            padding=self.padding,
            resample_filter=self.resample_filter,
            flip_weight=flip_weight,
            fused_modconv=fused_modconv,
            fmm_weights=fmm_weights,
            fmm_mod_type=self.fmm_mod_type,
            fmm_add_weight=self.fmm_add_weight,
            fmm_activation=self.fmm_activation,
            upsampling_mode=self.cfg.upsampling_mode,
            spatial_style=self.cfg.style.is_spatial and (not self.cfg.style.fallback_to_mean),
        )

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)

        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class ToRGBLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1, conv_clamp=None, channels_last=False, cfg={}):
        super().__init__()

        self.conv_clamp = conv_clamp
        self.cfg = OmegaConf.create(cfg)
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)

        memory_format = torch.channels_last if channels_last else torch.contiguous_format

        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

    def forward(self, x, w, fused_modconv=True, w_context: Optional[Tensor]=None, left_borders_idx=None):
        if self.cfg.style.is_spatial:
            styles = run_spatial_affine(self.affine, w, w_context, fallback_to_mean=self.cfg.style.fallback_to_mean)

            if not self.cfg.style.fallback_to_mean:
                fused_modconv = False
        else:
            styles = self.affine(w)

        styles = styles * self.weight_gain

        if self.cfg.style.is_spatial:
            fused_modconv = False

        x = modulated_conv2d(
            x=x,
            weight=self.weight,
            styles=styles,
            demodulate=False,
            fused_modconv=fused_modconv,
            spatial_style=self.cfg.style.is_spatial,
            left_borders_idx=left_borders_idx,
            w_lerp_kwargs=dict(
                w_coord_dist=self.cfg.patchwise.get('w_coord_dist', None),
                grid_size=self.cfg.patchwise.get('grid_size', None),
                w_lerp_multiplier=self.cfg.patchwise.get('w_lerp_multiplier', None),
            ),
        )

        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)

        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        out_channels,                       # Number of output channels.
        w_dim,                              # Intermediate latent (W) dimensionality.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of output color channels.
        is_last,                            # Is this the last block?
        architecture        = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        cfg                 = None,         # Synthezis config
        **layer_kwargs,                     # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()

        self.cfg = OmegaConf.create(cfg)
        self.in_channels = in_channels
        self.w_dim = w_dim

        if resolution <= self.cfg.min_resolution:
            self.resolution = self.cfg.min_resolution
            self.up = 1
            self.input_resolution = self.cfg.min_resolution
        else:
            self.resolution = resolution
            self.up = 2
            self.input_resolution = resolution // 2

        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0
        self.num_mod_coord_fusers = 0

        if in_channels == 0:
            self.input = GenInput(self.cfg.input, out_channels, w_dim)
            conv1_in_channels = self.input.channel_dim
            self.has_modulated_input = self.input.is_modulated
        else:
            conv1_in_channels = out_channels
            self.has_modulated_input = False

        kernel_size = self.cfg.coords.kernel_size if self.cfg.coords.enabled else 3
        synth_layer_class = PatchWiseSynthesisLayer if self.cfg.patchwise.enabled else SynthesisLayer

        if in_channels != 0:
            if self.cfg.coords.enabled:
                self.coords_fuser0 = ModulatedCoordFuser(self.cfg.coords, self.w_dim, self.resolution, self.use_fp16)
                self.num_mod_coord_fusers += (1 if self.coords_fuser0.is_modulated else 0)
                conv0_in_channels = in_channels + self.coords_fuser0.total_dim
            else:
                conv0_in_channels = in_channels

            synth_layer_kwargs_conv0 = dict(coord_dim_size=self.coords_fuser0.total_dim) if (self.cfg.patchwise.enabled and self.cfg.coords.enabled) else {}
            synth_layer_kwargs_conv1 = {}

            self.conv0 = synth_layer_class(
                conv0_in_channels,
                out_channels,
                w_dim=w_dim,
                resolution=self.resolution,
                kernel_size=kernel_size,
                up=self.up,
                resample_filter=resample_filter,
                conv_clamp=conv_clamp,
                channels_last=self.channels_last,
                cfg=cfg,
                **synth_layer_kwargs_conv0,
                **layer_kwargs)
            self.num_conv += 1
        else:
            if self.cfg.patchwise.enabled and self.cfg.coords.enabled and self.cfg.input.type == 'grid':
                synth_layer_kwargs_conv1 = dict(coord_dim_size=conv1_in_channels - out_channels)
            else:
                synth_layer_kwargs_conv1 = {}

        self.conv1 = synth_layer_class(
            conv1_in_channels,
            out_channels,
            w_dim=w_dim,
            resolution=self.resolution,
            kernel_size=kernel_size,
            conv_clamp=conv_clamp,
            channels_last=self.channels_last,
            cfg=cfg,
            **synth_layer_kwargs_conv1,
            **layer_kwargs)
        self.num_conv += 1

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(
                out_channels,
                img_channels,
                w_dim=w_dim,
                conv_clamp=conv_clamp,
                channels_last=self.channels_last,
                cfg=cfg)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
                up=self.up,
                resample_filter=resample_filter,
                channels_last=self.channels_last)

        self.num_modulated_transforms = self.num_conv + int(self.has_modulated_input) + self.num_torgb

        if self.is_last:
            self.num_ws = self.num_modulated_transforms
        else:
            # For non-last layers, we use a next-layer `w` to modulate them
            self.num_ws = self.num_modulated_transforms - self.num_torgb

    def progressive_growing_update(self, current_iteration: int):
        if self.cfg.coords.enabled and hasattr(self, 'coords_fuser0'):
            self.coords_fuser0.progressive_growing_update(current_iteration)

    def forward(self, x, img, ws, force_fp32=False, fused_modconv=None, ws_context=None, left_borders_idx=None, **layer_kwargs):
        misc.assert_shape(ws, [None, self.num_modulated_transforms, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        if ws_context is not None:
            w_context_iter = iter(ws_context.unbind(dim=2))
        dtype = torch.float16 if (self.use_fp16 and not force_fp32) else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        if fused_modconv is None:
            with misc.suppress_tracer_warnings(): # this value will be treated as a constant
                fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)

        # Input.
        if self.in_channels == 0:
            x = self.input(
                batch_size=ws.shape[0],
                w=(next(w_iter) if self.has_modulated_input else None),
                dtype=dtype,
                memory_format=memory_format,
                w_context=(None if not self.has_modulated_input or ws_context is None else next(w_context_iter)),
                left_borders_idx=left_borders_idx)
        else:
            misc.assert_shape(x, [None, self.in_channels, self.input_resolution, self.input_resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Coords fusion
        w_for_the_first_conv = next(w_iter)
        w_context_for_the_first_conv = None if ws_context is None else next(w_context_iter)

        if self.num_mod_coord_fusers > 0:
            # We are using the same w which is gonna be used for the upcoming convolution
            x = self.coords_fuser0(
                x,
                w_for_the_first_conv,
                left_borders_idx=left_borders_idx,
                dtype=dtype,
                memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(
                x,
                w_for_the_first_conv,
                fused_modconv=fused_modconv,
                w_context=w_context_for_the_first_conv,
                left_borders_idx=left_borders_idx,
                **layer_kwargs)
        elif self.architecture == 'resnet':
            assert left_borders_idx is None and ws_context is None
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, w_for_the_first_conv, fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
            x = self.conv0(
                x,
                w_for_the_first_conv,
                fused_modconv=fused_modconv,
                w_context=w_context_for_the_first_conv,
                left_borders_idx=left_borders_idx,
                **layer_kwargs)
            x = self.conv1(
                x,
                next(w_iter),
                fused_modconv=fused_modconv,
                w_context=(None if ws_context is None else next(w_context_iter)),
                left_borders_idx=left_borders_idx,
                **layer_kwargs)

        # ToRGB.
        if img is not None:
            misc.assert_shape(img, [None, self.img_channels, self.input_resolution, self.input_resolution])

            if self.up == 2:
                if self.cfg.patchwise.enabled:
                    img = patchwise_op(upfirdn2d.upsample2d, img, self.cfg.patchwise.grid_size, 2, self.resample_filter)
                else:
                    img = upfirdn2d.upsample2d(img, self.resample_filter)

        if self.is_last or self.architecture == 'skip':
            y = self.torgb(
                x,
                next(w_iter),
                fused_modconv=fused_modconv,
                w_context=(None if ws_context is None else next(w_context_iter)),
                left_borders_idx=left_borders_idx,
            )

            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32

        return x, img

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output image resolution.
        img_channels,               # Number of color channels.
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 0,        # Use FP16 for the N highest resolutions.
        cfg             = {},
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0

        super().__init__()

        self.cfg = OmegaConf.create(cfg)
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]

        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0

        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(
                in_channels,
                out_channels,
                w_dim=w_dim,
                resolution=res,
                img_channels=img_channels,
                is_last=is_last,
                use_fp16=use_fp16,
                cfg=cfg,
                **block_kwargs)

            self.num_ws += block.num_ws
            setattr(self, f'b{res}', block)

        if self.cfg.patchwise.enabled:
            self.register_buffer('max_shift_strength', torch.tensor([0.0]))

    def progressive_growing_update(self, current_iteration: int):
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            block.progressive_growing_update(current_iteration)

        if self.cfg.patchwise.enabled:
            shift_strength = min(current_iteration / self.cfg.style.kimg_to_reach_max_shift_strength, 1.0)
            self.max_shift_strength.copy_(torch.tensor([shift_strength]))

    def forward(self, ws, ws_context=None, left_borders_idx=None, **block_kwargs):
        block_ws = []
        block_ws_context = []

        if ws_context is not None:
            context_size = 2
            misc.assert_shape(ws_context, [ws.shape[0], context_size, self.num_ws, self.w_dim])

        if left_borders_idx is not None:
            left_borders_idx = (self.max_shift_strength * left_borders_idx.float()).to(torch.int32)

        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0

            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_modulated_transforms))
                if ws_context is None:
                    block_ws_context.append(None)
                else:
                    block_ws_context.append(ws_context.narrow(2, w_idx, block.num_modulated_transforms))

                w_idx += block.num_ws

        x = img = None

        for res, cur_ws, curr_ws_context in zip(self.block_resolutions, block_ws, block_ws_context):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, ws_context=curr_ws_context, left_borders_idx=left_borders_idx, **block_kwargs)

        return img

#----------------------------------------------------------------------------

@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                       # Input latent (Z) dimensionality.
        c_dim,                       # Conditioning label (C) dimensionality.
        w_dim,                       # Intermediate latent (W) dimensionality.
        img_resolution,              # Output resolution.
        img_channels,                # Number of output color channels.
        mapping_kwargs      = {},    # Arguments for MappingNetwork.
        synthesis_kwargs    = {},    # Arguments for SynthesisNetwork.
        synthesis_cfg       = {},    # Config for synthezis
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.synthesis_cfg = OmegaConf.create(synthesis_cfg)
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(
            w_dim=w_dim,
            img_resolution=img_resolution,
            img_channels=img_channels,
            cfg=synthesis_cfg,
            **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(
            z_dim=z_dim,
            c_dim=c_dim,
            w_dim=w_dim,
            num_ws=self.num_ws,
            num_modes=self.synthesis_cfg.num_modes,
            mode_max_norm=self.synthesis_cfg.mode_max_norm,
            **mapping_kwargs)

    def progressive_growing_update(self, *args, **kwargs):
        self.synthesis.progressive_growing_update(*args, **kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
        if self.synthesis_cfg.patchwise.enabled:
            modes_idx = torch.randint(low=0, high=self.synthesis_cfg.num_modes, size=(len(z),), device=z.device)
            ws_context = torch.stack([
                self.mapping(torch.randn_like(z), c, skip_w_avg_update=True, modes_idx=modes_idx),
                self.mapping(torch.randn_like(z), c, skip_w_avg_update=True, modes_idx=modes_idx),
            ], dim=1)
            w_dist = int(0.5 * self.synthesis_cfg.patchwise.w_coord_dist * self.synthesis_cfg.patchwise.grid_size)
            left_borders_idx = torch.randint(low=0, high=(2 * w_dist - self.synthesis_cfg.patchwise.grid_size), size=z.shape[:1], device=z.device)
        else:
            ws_context = None
            left_borders_idx = None
            modes_idx = None

        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, modes_idx=modes_idx)
        img = self.synthesis(ws, ws_context=ws_context, left_borders_idx=left_borders_idx, **synthesis_kwargs)

        return img

#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        tmp_channels,                       # Number of intermediate channels.
        out_channels,                       # Number of output channels.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of input color channels.
        first_layer_idx,                    # Index of the first layer.
        architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
        activation          = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        freeze_layers       = 0,            # Freeze-D: Number of layers to freeze.
        is_modulated: bool  = False,        # Should we modulate convolutions?
        c_dim: int          = 0,            # What is c_dim in case we want to modulate them?
    ):
        assert in_channels in [0, tmp_channels]
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.c_dim = c_dim

        self.num_layers = 0
        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable
        trainable_iter = trainable_gen()

        if in_channels == 0 or architecture == 'skip':
            self.fromrgb = Conv2dLayer(
                img_channels,
                tmp_channels,
                kernel_size=1,
                activation=activation,
                trainable=next(trainable_iter),
                conv_clamp=conv_clamp,
                channels_last=self.channels_last)

        if is_modulated and c_dim > 0:
            assert next(trainable_iter)
            self.conv0 = SynthesisLayer(
                tmp_channels,
                tmp_channels,
                w_dim=c_dim,
                resolution=self.resolution,
                kernel_size=3,
                activation=activation,
                use_noise=False,
                conv_clamp=conv_clamp,
                channels_last=self.channels_last)

            assert next(trainable_iter)
            self.conv1 = SynthesisLayer(
                tmp_channels,
                out_channels,
                w_dim=c_dim,
                resolution=self.resolution,
                kernel_size=3,
                activation=activation,
                use_noise=False,
                conv_clamp=conv_clamp,
                channels_last=self.channels_last,
                down=2)
        else:
            self.conv0 = Conv2dLayer(
                tmp_channels,
                tmp_channels,
                kernel_size=3,
                activation=activation,
                trainable=next(trainable_iter),
                conv_clamp=conv_clamp,
                channels_last=self.channels_last)

            self.conv1 = Conv2dLayer(
                tmp_channels,
                out_channels,
                kernel_size=3,
                activation=activation,
                down=2,
                trainable=next(trainable_iter),
                resample_filter=resample_filter,
                conv_clamp=conv_clamp,
                channels_last=self.channels_last)

        if architecture == 'resnet':
            self.skip = Conv2dLayer(
                tmp_channels,
                out_channels,
                kernel_size=1,
                bias=False,
                down=2,
                trainable=next(trainable_iter),
                resample_filter=resample_filter,
                channels_last=self.channels_last)

    def forward(self, x, img, c=None, force_fp32=False):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        if x is not None:
            misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0 or self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            img = upfirdn2d.downsample2d(img, self.resample_filter) if self.architecture == 'skip' else None

        # Main layers.
        if self.architecture == 'resnet':
            gain = np.sqrt(0.5)
            y = self.skip(x, gain=gain)
            x = self.conv0(x, c) if self.c_dim > 0 else self.conv0(x)
            x = self.conv1(x, c, gain=gain) if self.c_dim > 0 else self.conv1(x, gain=gain)
            x = y.add_(x)
        else:
            x = self.conv0(x, c) if self.c_dim > 0 else self.conv0(x)
            x = self.conv1(x, c) if self.c_dim > 0 else self.conv1(x)

        assert x.dtype == dtype
        return x, img

#----------------------------------------------------------------------------

@persistence.persistent_class
class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with misc.suppress_tracer_warnings(): # as_tensor results are registered as constants
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2,3,4])             # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.

        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        cmap_dim,                       # Dimensionality of mapped conditioning label, 0 = no label.
        resolution,                     # Resolution of this block.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation          = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
        self.out = FullyConnectedLayer(in_channels, 1 if cmap_dim == 0 else cmap_dim)

    def forward(self, x, img, cmap, force_fp32=False):
        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution]) # [NCHW]
        _ = force_fp32 # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)

        # Conditioning.
        if self.cmap_dim > 0:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        assert x.dtype == dtype
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class Discriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 0,        # Use FP16 for the N highest resolutions.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
    ):
        super().__init__()

        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]

        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0

        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)

            block = DiscriminatorBlock(
                in_channels,
                tmp_channels,
                out_channels,
                resolution=res,
                first_layer_idx=cur_layer_idx,
                use_fp16=use_fp16,
                **block_kwargs,
                **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers

        if c_dim > 0:
            self.mapping = MappingNetwork(
                z_dim=0,
                c_dim=c_dim,
                w_dim=cmap_dim,
                num_ws=None,
                w_avg_beta=None,
                **mapping_kwargs)

        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)

    def forward(self, img, c, **block_kwargs):
        x = None

        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, c=c, **block_kwargs)

        cmap = None

        if self.c_dim > 0:
            cmap = self.mapping(None, c)

        x = self.b4(x, img, cmap)

        return x

#----------------------------------------------------------------------------
