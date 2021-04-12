from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf, DictConfig

from torch_utils import persistence
from torch_utils.ops import bias_act
from torch_utils import misc
from torch_utils.ops.fast_bilinear_mult import fast_manual_bilinear_mult_row

#----------------------------------------------------------------------------

@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias

        if b is not None:
            b = b.to(x.dtype)

            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)

        return x


#----------------------------------------------------------------------------

@persistence.persistent_class
class GenInput(nn.Module):
    def __init__(self, cfg: DictConfig, channel_dim: int, w_dim: int):
        super().__init__()

        self.cfg = cfg

        if self.cfg.type == 'multi_modal':
            self.input = MultiModalInput(channel_dim, self.cfg.resolution, w_dim, **self.cfg.kwargs)
            self.channel_dim = channel_dim
        elif self.cfg.type == 'const':
            self.input = torch.nn.Parameter(torch.randn([channel_dim, self.cfg.resolution, self.cfg.resolution]))
            self.channel_dim = channel_dim
        elif self.cfg.type == 'periodic_const':
            self.input = PeriodicConstInput(channel_dim, self.cfg.resolution)
            self.channel_dim = channel_dim
        elif self.cfg.type == 'grid':
            self.input = GridInput(channel_dim, self.cfg.resolution, w_dim, **self.cfg.kwargs)
            self.channel_dim = self.input.get_channel_dim()
        elif self.cfg.type == 'coords':
            self.input = CoordsInput(self.cfg.resolution, **self.cfg.kwargs)
            self.channel_dim = self.input.get_channel_dim()
        elif self.cfg.type == 'modulated':
            self.input = ModulatedInput(channel_dim, self.cfg.resolution, w_dim)
            self.channel_dim = channel_dim
        elif self.cfg.type == 'coord_noise':
            self.input = CoordNoiseInput(channel_dim, self.cfg.resolution, **self.cfg.kwargs)
            self.channel_dim = self.input.get_channel_dim()
        else:
            raise NotImplementedError

        self.is_modulated = self.cfg.type in ('multi_modal', 'modulated', 'grid')

    def forward(self, batch_size: int, w: Tensor=None, dtype=None, memory_format=None, w_context=None, left_borders_idx=None) -> Tensor:
        if self.cfg.type == 'multi_modal':
            x = self.input(w).to(dtype=dtype, memory_format=memory_format)
        elif self.cfg.type == 'const':
            x = self.input.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([batch_size, 1, 1, 1])
        elif self.cfg.type == 'periodic_const':
            x = self.input(batch_size)
            x = x.to(dtype=dtype, memory_format=memory_format)
        elif self.cfg.type == 'grid':
            x = self.input(batch_size, w, w_context, left_borders_idx)
            x = x.to(dtype=dtype, memory_format=memory_format)
        elif self.cfg.type == 'coords':
            x = self.input(batch_size).to(dtype=dtype, memory_format=memory_format)
        elif self.cfg.type == 'modulated':
            x = self.input(w).to(dtype=dtype, memory_format=memory_format)
        elif self.cfg.type == 'coord_noise':
            x = self.input(batch_size, left_borders_idx)
            x = x.to(dtype=dtype, memory_format=memory_format)
        else:
            raise NotImplementedError

        return x


#----------------------------------------------------------------------------

@persistence.persistent_class
class MultiModalInput(nn.Module):
    def __init__(self,
            channel_dim: int,
            resolution: int,
            w_dim: int,
            num_groups: int,
            num_modes: int,
            demodulate: bool,
            temperature: float):

        super().__init__()

        assert channel_dim % num_groups == 0

        self.num_groups = num_groups
        self.num_modes = num_modes
        self.num_params = num_groups * num_modes
        self.resolution = resolution
        self.channel_dim = channel_dim
        self.demodulate = demodulate
        self.temperature = temperature
        self.inputs = nn.Parameter(torch.randn(1, num_groups, num_modes, channel_dim // num_groups, resolution, resolution))
        self.affine = FullyConnectedLayer(w_dim, self.num_params, bias_init=0)

    def forward(self, w: Tensor) -> Tensor:
        styles = self.affine(w) # [batch_size, num_groups * num_modes]
        probs = (styles.view(batch_size, self.num_groups, self.num_modes) / self.temperature).softmax(dim=1)
        probs = probs.view(batch_size, self.num_groups, self.num_modes, 1, 1, 1)
        inputs = (self.inputs * probs).sum(dim=2) # [batch_size, num_groups, channel_dim // num_groups, resolution, resolution]
        inputs = inputs.view(batch_size, self.channel_dim, self.resolution, self.resolution)

        if self.demodulate:
            inputs = inputs / inputs.norm(float('inf'), dim=[2, 3], keepdim=True)

        return inputs


@persistence.persistent_class
class ModulatedInput(nn.Module):
    def __init__(self, channel_dim: int, resolution: int, w_dim: int, demodulate: bool=True):
        super().__init__()

        self.const_input = torch.nn.Parameter(torch.randn([channel_dim, resolution, resolution]))
        self.affine = FullyConnectedLayer(w_dim, channel_dim, bias_init=1)
        self.channel_dim = channel_dim
        self.demodulate = demodulate

    def forward(self, w: Tensor) -> Tensor:
        styles = self.affine(w) # [batch_size, channel_dim]
        x = self.const_input * style.view(w.size(0), self.channel_dim, 1, 1)

        if self.demodulate:
            x = x * (x.square().sum(dim=[1,2,3]) + 1e-8).rsqrt()

        return x


@persistence.persistent_class
class CoordsInput(nn.Module):
    def __init__(self, resolution: int, **basis_gen_kwargs):
        super().__init__()

        batch_size = 1
        raw_coords = generate_coords(1, resolution)
        basis = generate_logarithmic_basis(resolution, **basis_gen_kwargs) # [dim, 2]
        basis = basis.unsqueeze(0) # [1, dim, 2]
        coord_embs = torch.einsum('bdc,bcxy->bdxy', basis, raw_coords).sin() # [batch_size, dim, img_size, img_size]

        self.register_buffer('coord_embs', coord_embs) # [batch_size, dim, img_size, img_size]
        self.coord_embs_cache = None

    def get_channel_dim(self) -> int:
        return self.coord_embs.shape[1]

    def forward(self, batch_size: int) -> Tensor:
        if (self.coord_embs_cache is None) or batch_size != self.coord_embs_cache.shape[0]:
            self.coord_embs_cache = self.coord_embs.repeat(batch_size, 1, 1, 1)
            self.coord_embs_cache = self.coord_embs_cache.contiguous()

        return self.coord_embs_cache


@persistence.persistent_class
class CoordNoiseInput(nn.Module):
    def __init__(self, channel_dim: int, resolution: int, coords_cfg={}):
        super().__init__()

        self.channel_dim = channel_dim
        self.resolution = resolution
        self.coord_fuser = ModulatedCoordFuser(OmegaConf.create(coords_cfg), w_dim=None, resolution=resolution, use_fp16=False)

    def get_channel_dim(self) -> int:
        return self.channel_dim + self.coord_fuser.compute_total_dim()

    def forward(self, batch_size: int, left_borders_idx: Tensor) -> Tensor:
        misc.assert_shape(left_borders_idx, [batch_size])

        noise = torch.randn(batch_size, self.channel_dim, self.resolution, self.resolution, device=left_borders_idx.device)
        out = self.coord_fuser(noise, left_borders_idx=left_borders_idx, memory_format=torch.contiguous_format)

        return out


@persistence.persistent_class
class PeriodicConstInput(nn.Module):
    """
    It is like constant input, but periodic
    """
    def __init__(self, channel_dim: int, resolution: int):
        super().__init__()

        self.resolution = resolution
        self.const_input = torch.nn.Parameter(torch.randn([channel_dim, resolution, resolution]))

    def forward(self, batch_size: int, shifts: Optional[Tensor]=None) -> Tensor:
        x = self.const_input.unsqueeze(0).repeat([batch_size, 1, 1, 1]) # [b, c, h, w]

        if shifts is not None:
            misc.assert_shape(shifts, [batch_size, 2])
            assert shifts.max().item() <= 1.0
            assert shifts.min().item() >= -1.0

            coords = generate_coords(batch_size, self.const_input.shape[1], device=x.device, align_corners=True) # [b, 2, h, w]

            # # Applying the shift
            # coords = coords + shifts.view(batch_size, 2, 1, 1) # [b, 2, h, w]

            # # Converting into F.grid_sample coords:
            # # 1. Convert the range
            # coords = coords + 1 # [-1, 1] => [0, 2]
            # # 2. Perform padding_mode=replicate
            # # coords[coords > 0] = coords[coords > 0] % (2 + 1e-12)
            # # coords[coords < 0] = -(-coords[coords < 0] % 2) + 2 + (1e-12)
            # # 3. Convert back to [-1, 1] range
            # coords = coords - 1 # [0, 2] => [-1, 1]
            # # 4. F.grid_sample uses flipped coordinates (TODO: should we too?)
            # coords[:, 1] = coords[:, 1] * -1.0
            # # 5. It also uses different shape
            # coords = coords.permute(0, 2, 3, 1) # [b, h, w, 2]

            # Performing a slower, but less error-prone approach
            # (convert shifts from [-1, 1] to [-2, 2], so we are now [-3, 3])
            coords = coords + 2 * shifts.view(batch_size, 2, 1, 1) # [b, 2, h, w]
            coords = coords / 3 # [-3, 3] => [-1, 1] range
            coords = coords.permute(0, 2, 3, 1)
            assert coords.min().item() >= -1
            assert coords.max().item() <= 1

            x = torch.cat([x, x, x], dim=3) # [b, c, h, w * 3]
            x = F.grid_sample(x, coords, mode='bilinear', align_corners=True) # [b, c, h, w]

            # torch.save(coords.detach().cpu(), '/tmp/trash/coords')
            # torch.save(x.detach().cpu(), '/tmp/trash/x')
            # torch.save(self.const_input.detach().cpu(), '/tmp/trash/const_input')

            # assert torch.allclose(x[0], self.const_input, atol=1e-4)

        return x


@persistence.persistent_class
class GridInput(nn.Module):
    """
    For COCO-GAN, our input is grid-like and consists on 2 things:
        - learnable coordinates
        - high-frequency (up to 1 image for the whole period)
    """
    def __init__(self, channel_dim: int, resolution: int, w_dim: int, grid_size: int, w_coord_dist: int, w_lerp_multiplier: bool):
        super().__init__()

        self.resolution = resolution
        self.grid_size = grid_size
        self.channel_dim = channel_dim
        self.w_dim = w_dim
        self.w_lerp_multiplier = w_lerp_multiplier

        # Distance between `w` and `w_after` measured in number of steps
        # By default, it equals to grid_size, but should be increased I guess
        self.w_coord_dist = w_coord_dist

        # # Learnable patch coordinate embeddings
        # self.const_input = torch.nn.Parameter(torch.randn([channel_dim // 2, resolution, resolution]))
        self.input_column = torch.nn.Parameter(torch.randn([channel_dim, resolution]))

        # Predictable patch embeddings
        self.affine = FullyConnectedLayer(w_dim, channel_dim, bias_init=0)

        # Fixed coordinate patch embeddings
        self.register_buffer('basis', generate_logarithmic_basis(
            resolution,
            self.channel_dim,
            remove_lowest_freq=True,
            use_diagonal=True)) # [dim, 2]

    def get_channel_dim(self) -> int:
        return self.basis.shape[0] * 2 + self.channel_dim

    def forward(self, batch_size: int, w: Tensor, w_context: Tensor, left_borders_idx: Tensor) -> Tensor:
        misc.assert_shape(w, [batch_size, self.w_dim])
        misc.assert_shape(w_context, [batch_size, 2, self.w_dim])
        misc.assert_shape(left_borders_idx, [batch_size])

        # Computing the global features
        w_all = torch.stack([w_context[:, 0], w, w_context[:, 1]], dim=1) # [b, 3, w_dim]
        styles = self.affine(w_all.view(-1, self.w_dim)).view(batch_size, 3, self.channel_dim) # [b, 2, c]
        raw_const_inputs = self.input_column.unsqueeze(0).unsqueeze(3).repeat(batch_size, 1, 1, self.resolution) # [b, c, h, w]
        latents = fast_manual_bilinear_mult_row(raw_const_inputs, styles, left_borders_idx, self.grid_size, self.w_coord_dist, self.w_lerp_multiplier)

        # Ok, now for each cell in the grid we need to compute its high-frequency coordinates
        # Otherwise, it will be too difficult for the model to understand the relative positions
        coords = generate_shifted_coords(left_borders_idx, self.resolution, self.grid_size, self.w_coord_dist, device=w.device)
        bases = self.basis.unsqueeze(0).repeat(batch_size, 1, 1) # [batch_size, dim, 2]
        raw_coord_embs = torch.einsum('bdc,bcxy->bdxy', bases, coords) # [batch_size, dim, img_size, img_size]
        coord_embs = torch.cat([raw_coord_embs.sin(), raw_coord_embs.cos()], dim=1) # [batch_size, dim * 2, img_size, img_size]

        # Computing final inputs
        inputs = torch.cat([latents, coord_embs], dim=1) # [b, c, grid_size, grid_size]

        return inputs


@persistence.persistent_class
class ModulatedCoordFuser(nn.Module):
    """
    CoordFuser which concatenates coordinates across dim=1 (we assume channel_first format)
    """
    def __init__(self, cfg: DictConfig, w_dim: int, resolution: int, use_fp16: bool):
        super().__init__()

        self.cfg = cfg
        self.dim = self.cfg.channels_dict[str(resolution)].dim
        self.use_cosine = self.cfg.channels_dict[str(resolution)].use_cosine
        self.W_size = self.dim * self.cfg.coord_dim
        self.b_size = self.dim
        self.resolution = resolution

        if self.cfg.logarithmic:
            self.register_buffer('basis', generate_logarithmic_basis(
                resolution,
                self.dim,
                remove_lowest_freq=self.cfg.remove_lowest_freq,
                use_diagonal=self.cfg.channels_dict[str(resolution)].use_diagonal)) # [dim, 2]
            self.coord_embs_cache = None

            if self.cfg.growth_schedule.enabled:
                self.final_img_resolution = self.cfg.growth_schedule['final_img_resolution'] # TODO(universome): do not hardcode...
                assert resolution <= self.final_img_resolution
                self.num_freqs = int(np.ceil(np.log2(resolution)))
                self.register_buffer('growth_weights', torch.zeros(self.num_freqs)) # [num_freqs]
                self.progressive_growing_update(0)

                # Here we use "* 4" because we have 1) vertical, 2) horizontal,
                # 3) main-diagonal, 4) anti-diagonal wavefronts
                assert self.num_freqs * 4 <= self.dim
            else:
                self.growth_weights = None
        else:
            assert not self.cfg.growth_schedule.enabled

            if not self.cfg.no_fourier_fallback:
                self.affine = FullyConnectedLayer(w_dim, self.W_size + self.b_size, bias_init=0)

        if self.cfg.use_cips_embs > 0:
            dtype = torch.fp16 if use_fp16 else torch.float32
            self.cips_embs = nn.Parameter(torch.randn(1, self.dim, resolution, resolution).to(dtype).contiguous())

        self.total_dim = self.compute_total_dim()
        self.is_modulated = False if (self.cfg.fallback or self.cfg.no_fourier_fallback) else True

    def compute_total_dim(self) -> int:
        if self.cfg.fallback: return 0
        total_dim = 0

        if self.cfg.no_fourier_fallback:
            total_dim += self.cfg.coord_dim
        elif self.cfg.logarithmic:
            if self.use_cosine:
                total_dim += self.basis.shape[0] * 2
            else:
                total_dim += self.basis.shape[0]
        else:
            if self.use_cosine:
                total_dim += self.dim * 2
            else:
                total_dim += self.dim

        if self.cfg.use_cips_embs:
            total_dim += self.dim

        return total_dim

    def progressive_growing_update(self, current_iteration: int):
        if self.cfg.growth_schedule.enabled:
            # TODO(universome): fix this
            num_freqs = np.ceil(np.log2(self.final_img_resolution)).astype(int)
            common_args = (self.cfg.growth_schedule.time_to_reach_all, num_freqs)
            # We use (j-1) here instead of j as in the nerfies paper
            # because they use (x,y) linear input for their coordinate embeddings and we don't
            # Actually, maybe we should...
            # So, in this way we want our lowest frequency to be always enabled
            growth_weights = [compute_freq_weight(current_iteration, j - 1, *common_args) for j in range(self.growth_weights.shape[0])] # [num_freqs]

            self.growth_weights.copy_(torch.tensor(growth_weights))

    def forward(self, x: Tensor, w: Tensor=None, left_borders_idx: Tensor=None, dtype=None, memory_format=None) -> Tensor:
        """
        Dims:
            @arg x is [batch_size, in_channels, img_size, img_size]
            @arg w is [batch_size, w_dim]
            @return out is [batch_size, in_channels + fourier_dim + cips_dim, img_size, img_size]
        """
        assert memory_format is torch.contiguous_format

        if self.cfg.fallback:
            return x

        batch_size, in_channels, img_size = x.shape[:3]

        if left_borders_idx is not None:
            raw_coords = generate_shifted_coords(left_borders_idx, img_size, self.cfg.grid_size, self.cfg.w_coord_dist, device=x.device)
        else:
            raw_coords = generate_coords(batch_size, img_size, x.device) # [batch_size, coord_dim, img_size, img_size]

        if self.cfg.no_fourier_fallback:
            coord_embs = raw_coords
        elif self.cfg.logarithmic:
            if (not self.cfg.growth_schedule.enabled) \
                or (self.coord_embs_cache is None) \
                or (self.coord_embs_cache.shape != (batch_size, 2, self.basis.shape[0])) \
                or (self.coord_embs_cache.device != x.device):

                if self.cfg.growth_schedule.enabled:
                    growth_weights = self.growth_weights.repeat(4) # [0,1,2] => [0,1,2,0,1,2,...]
                    basis = self.basis * growth_weights.unsqueeze(1) # [dim, 2]
                else:
                    basis = self.basis # [dim, 2]

                bases = basis.unsqueeze(0).repeat(batch_size, 1, 1) # [batch_size, dim, 2]
                raw_coord_embs = torch.einsum('bdc,bcxy->bdxy', bases, raw_coords) # [batch_size, dim, img_size, img_size]

                if self.use_cosine:
                    self.coord_embs_cache = torch.cat([raw_coord_embs.sin(), raw_coord_embs.cos()], dim=1) # [batch_size, dim * 2, img_size, img_size]
                else:
                    self.coord_embs_cache = raw_coord_embs.sin()

                self.coord_embs_cache = self.coord_embs_cache.contiguous()

            coord_embs = self.coord_embs_cache
        else:
            mod = self.affine(w) # [batch_size, W_size + b_size]
            W = self.cfg.fourier_scale * mod[:, :self.W_size] # [batch_size, W_size]
            W = W.view(batch_size, self.dim, self.cfg.coord_dim) # [batch_size, fourier_dim, coord_dim]
            bias = mod[:, self.W_size:].view(batch_size, self.dim, 1, 1) # [batch_size, fourier_dim, 1]
            raw_coord_embs = (torch.einsum('bdc,bcxy->bdxy', W, raw_coords) + bias) # [batch_size, coord_dim, img_size, img_size]

            if self.use_cosine:
                coord_embs = torch.cat([raw_coord_embs.sin(), raw_coord_embs.cos()], dim=1) # [batch_size, dim * 2, img_size, img_size]
            else:
                coord_embs = raw_coord_embs.sin()

        coord_embs = coord_embs.to(dtype=dtype, memory_format=memory_format)
        out = torch.cat([x, coord_embs], dim=1) # [batch_size, in_channels + fourier_dim, img_size, img_size]

        if self.cfg.use_cips_embs > 0:
            cips_embs = self.cips_embs.repeat([batch_size, 1, 1, 1])
            cips_embs = cips_embs.to(dtype=dtype, memory_format=memory_format)
            out = torch.cat([out, cips_embs], dim=1) # [batch_size, in_channels + fourier_dim + cips_emb, img_size, img_size]

        return out


def fmm_modulate(
    conv_weight: Tensor,
    fmm_weights: nn.Module,
    fmm_mod_type: str='mult',
    demodulate: bool=False,
    fmm_add_weight: float=1.0,
    activation: Optional[str]=None) -> Tensor:
    """
    Applies FMM fmm_weights to a given conv weight tensor
    """
    batch_size, out_channels, in_channels, kh, kw = conv_weight.shape

    assert fmm_weights.shape[1] % (in_channels + out_channels) == 0

    rank = fmm_weights.shape[1] // (in_channels + out_channels)
    lhs = fmm_weights[:, : rank * out_channels].view(batch_size, out_channels, rank)
    rhs = fmm_weights[:, rank * out_channels :].view(batch_size, rank, in_channels)

    modulation = lhs @ rhs # [batch_size, out_channels, in_channels]
    modulation = modulation / np.sqrt(rank)
    misc.assert_shape(modulation, [batch_size, out_channels, in_channels])
    modulation = modulation.unsqueeze(3).unsqueeze(4) # [batch_size, out_channels, in_channels, 1, 1]

    if activation == "tanh":
        modulation = modulation.tanh()
    elif activation in ['linear', None]:
        pass
    elif activation == 'sigmoid':
        modulation = modulation.sigmoid() - 0.5
    else:
        raise NotImplementedError

    if fmm_mod_type == 'mult':
        out = conv_weight * (modulation + 1.0)
    elif fmm_mod_type == 'add':
        out = conv_weight + fmm_add_weight * modulation
    else:
        raise NotImplementedError

    if demodulate:
        out = out / out.norm(dim=[2, 3, 4], keepdim=True)

    return out


def generate_coords(batch_size: int, img_size: int, device='cpu', align_corners: bool=False) -> Tensor:
    """
    Generates the coordinates in [-1, 1] range for a square image
    if size (img_size x img_size) in such a way that
    - upper left corner: coords[0, 0] = (-1, -1)
    - upper right corner: coords[img_size - 1, img_size - 1] = (1, 1)
    """
    if align_corners:
        row = torch.linspace(-1, 1, img_size, device=device).float() # [img_size]
    else:
        row = (torch.arange(0, img_size, device=device).float() / img_size) * 2 - 1 # [img_size]
    x_coords = row.view(1, -1).repeat(img_size, 1) # [img_size, img_size]
    y_coords = x_coords.t().flip(dims=(0,)) # [img_size, img_size]

    coords = torch.stack([x_coords, y_coords], dim=2) # [img_size, img_size, 2]
    coords = coords.view(-1, 2) # [img_size ** 2, 2]
    coords = coords.t().view(1, 2, img_size, img_size).repeat(batch_size, 1, 1, 1) # [batch_size, 2, img_size, img_size]

    return coords


def generate_logarithmic_basis(
    resolution: int,
    max_num_feats: int=np.float('inf'),
    remove_lowest_freq: bool=False,
    use_diagonal: bool=True) -> Tensor:
    """
    Generates a directional logarithmic basis with the following directions:
        - horizontal
        - vertical
        - main diagonal
        - anti-diagonal
    """
    max_num_feats_per_direction = np.ceil(np.log2(resolution)).astype(int)
    bases = [
        generate_horizontal_basis(max_num_feats_per_direction),
        generate_vertical_basis(max_num_feats_per_direction),
    ]

    if use_diagonal:
        bases.extend([
            generate_diag_main_basis(max_num_feats_per_direction),
            generate_anti_diag_basis(max_num_feats_per_direction),
        ])

    if remove_lowest_freq:
        bases = [b[1:] for b in bases]

    # If we do not fit into `max_num_feats`, then trying to remove the features in the order:
    # 1) anti-diagonal 2) main-diagonal
    while (max_num_feats_per_direction * len(bases) > max_num_feats) and (len(bases) > 2):
        bases = bases[:-1]

    basis = torch.cat(bases, dim=0)

    # If we still do not fit, then let's remove each second feature,
    # then each third, each forth and so on
    # We cannot drop the whole horizontal or vertical direction since otherwise
    # model won't be able to locate the position
    # (unless the previously computed embeddings encode the position)
    # while basis.shape[0] > max_num_feats:
    #     num_exceeding_feats = basis.shape[0] - max_num_feats
    #     basis = basis[::2]

    assert basis.shape[0] <= max_num_feats, \
        f"num_coord_feats > max_num_fixed_coord_feats: {basis.shape, max_num_feats}."

    return basis


def generate_horizontal_basis(num_feats: int) -> Tensor:
    return generate_wavefront_basis(num_feats, [0.0, 1.0], 4.0)


def generate_vertical_basis(num_feats: int) -> Tensor:
    return generate_wavefront_basis(num_feats, [1.0, 0.0], 4.0)


def generate_diag_main_basis(num_feats: int) -> Tensor:
    return generate_wavefront_basis(num_feats, [-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], 4.0 * np.sqrt(2))


def generate_anti_diag_basis(num_feats: int) -> Tensor:
    return generate_wavefront_basis(num_feats, [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], 4.0 * np.sqrt(2))


def generate_wavefront_basis(num_feats: int, basis_block: List[float], period_length: float) -> Tensor:
    period_coef = 2.0 * np.pi / period_length
    basis = torch.tensor([basis_block]).repeat(num_feats, 1) # [num_feats, 2]
    powers = torch.tensor([2]).repeat(num_feats).pow(torch.arange(num_feats)).unsqueeze(1) # [num_feats, 1]
    result = basis * powers * period_coef # [num_feats, 2]

    return result.float()


@torch.no_grad()
def generate_random_coords_shifts(batch_size: int, period_length: float=4.0, device: str='cpu') -> Tensor:
    """
    Generates shift the coordinates 1 half-period to the right
    Our half-period occupies the range [-1, 1]
    To shift it to [1, 3] we need to add U[-1, 1] * 0.5 * period_length to each coordinates set
    So, it just generates a random array of 0/2 values, like [0,0,2,2,0,2,...]
    """
    horizontal_shifts = (torch.rand(batch_size, device=device) - 0.5) * 2.0 * (0.5 * period_length) # [batch_size]
    vertical_shifts = torch.zeros(batch_size, device=device) # Do not shift vertical coordinates

    shifts = torch.cat([
        horizontal_shifts.unsqueeze(1),
        vertical_shifts.unsqueeze(1),
    ], dim=1).unsqueeze(2).unsqueeze(3).contiguous() # [batch_size, 2, 1, 1]

    return shifts


@torch.no_grad()
def compute_freq_weight(iteration: int, freq_idx: int, time_to_reach_all: int, total_num_freqs: int) -> float:
    progress_alpha: float = total_num_freqs * iteration / time_to_reach_all
    weight = (1.0 - np.cos(np.pi * np.clip(progress_alpha - freq_idx, 0, 1))) * 0.5

    return weight


def generate_shifted_coords(left_borders_idx: Tensor, img_size: int, grid_size: int, w_coord_dist: float, device='cpu') -> Tensor:
    coords = generate_coords(len(left_borders_idx), img_size, device=device) # [b, 2, grid_size, grid_size]

    # We need to convert left_borders_idx to coordinates shifts, knowing that
    # the relative unshifted coordinate position of the left border is -1.0 (i.e. at w_left)
    patch_size = img_size // grid_size
    w_dist = int(0.5 * w_coord_dist * img_size) # distance in pixels
    left_border_rel_pos = (left_borders_idx.to(torch.float32) * patch_size - w_dist) / w_dist # in [-1, 1 - grid_size/w_dist] range
    shifts = left_border_rel_pos * w_coord_dist # [batch_size]

    # Add +1 to account for left_border => center
    shifts = shifts + 1.0

    # Finally, shift the x-axis
    coords[:, 0] = coords[:, 0] + shifts.unsqueeze(1).unsqueeze(2) # [b, grid_size, grid_size]

    return coords
