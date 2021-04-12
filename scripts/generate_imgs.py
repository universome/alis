import os
from PIL import Image
import argparse
from tqdm import tqdm
import torch
import torchvision.transforms.functional as TVF
import numpy as np
from distutils.dir_util import copy_tree

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--exp_name', required=True, type=str, help='Name of the experiment')
parser.add_argument('-bs', '--batch_size', default=100, type=int)
parser.add_argument('-n', '--num_frames', default=50000, type=int)
parser.add_argument('--sample_normal', action='store_true')
parser.add_argument('--sample_infinite', action='store_true')
parser.add_argument('--ckpt_iter', type=int, help='Checkpoint iteration')
# parser.add_argument('-d', '--log_folder', required=True, type=str, help='Path to all the model data')
# parser.add_argument('-d', '--save_dir', required=True, type=str, help='Where to save the images')
args = parser.parse_args()

exps_dir = '/ibex/scratch/projects/c2112/new-sgf-experiments'
run_dir = [f for f in os.listdir(f'{exps_dir}/{args.exp_name}') if f.startswith('00000-')][0]
# log_folder = 'locogan_siren_gs_4_dist2_tower_0.7-f6a558a/00000-tower_train_t_0.7-mirror-auto4-noaug'

# exp_name, run_dir = args.log_folder.split('/')
project_dir = f'{exps_dir}/{args.exp_name}'
os.chdir(project_dir)

if os.path.isdir(f'{project_dir}/scripts'):
    copy_tree(f'{project_dir}/scripts', f'{project_dir}/custom_scripts')

# import sys; sys.path.extend(['.', '..'])
import sys; sys.path.append(f'/ibex/scratch/projects/c2112/new-sgf-experiments/{args.exp_name}')

from training.networks import SynthesisLayer
from training.networks import PatchWiseSynthesisLayer
import dnnlib
from custom_scripts.legacy import load_network_pkl

torch.set_grad_enabled(False)

network_pkls = sorted([f for f in os.listdir(f'{project_dir}/{run_dir}') if f.endswith('.pkl')])
if args.ckpt_iter is None:
    network_pkl = f'{project_dir}/{run_dir}/{network_pkls[-1]}'
else:
    network_pkl = f'{project_dir}/{run_dir}/network-snapshot-{args.ckpt_iter:06d}.pkl'
print(f'Loading {network_pkls[-1]} checkpoint')
device = 'cuda'

with dnnlib.util.open_url(network_pkl) as f:
    G = load_network_pkl(f)['G_ema'].to(device) # type: ignore
    G.eval()
    G.progressive_growing_update(100000)

for res in G.synthesis.block_resolutions:
    block = getattr(G.synthesis, f'b{res}')
    if hasattr(block, 'conv0'):
        block.conv0.use_noise = False
    block.conv1.use_noise = False


batch_size = args.batch_size
num_frames = args.num_frames
assert num_frames % batch_size == 0
num_steps = num_frames // batch_size

if args.sample_normal:
    save_dir = f'/tmp/samples/{args.exp_name}/normal_samples'
    os.makedirs(save_dir, exist_ok=True)

    for i in tqdm(range(num_steps), desc=f'Generating normal images for {args.exp_name}'):
        z = torch.randn(batch_size, G.z_dim).to(device)
        imgs = (G(z, c=None).clamp(-1, 1) * 0.5 + 0.5).cpu()

        for j, img in enumerate(imgs):
            img = TVF.to_pil_image(img)
            img.save(f'{save_dir}/{i * args.batch_size + j:06d}.png')


if args.sample_infinite:
    save_dir = f'/tmp/samples/{args.exp_name}/infinite_samples'
    os.makedirs(save_dir, exist_ok=True)

    # num_frames_per_w = G.synthesis_cfg.patchwise.w_coord_dist // 2
    # w_range = 2 * num_frames_per_w * G.synthesis_cfg.patchwise.grid_size
    # max_shift = (num_frames_per_w * 2 - 1) * G.synthesis_cfg.patchwise.grid_size

    z_l = torch.randn(batch_size, G.z_dim, device=device)
    z_c = torch.randn_like(z_l)
    z_r = torch.randn_like(z_l)

    modes_idx_l = torch.randint(0, G.synthesis_cfg.num_modes, size=(batch_size,), device=device)
    modes_idx_c = torch.randint(0, G.synthesis_cfg.num_modes, size=(batch_size,), device=device)
    modes_idx_r = torch.randint(0, G.synthesis_cfg.num_modes, size=(batch_size,), device=device)

    w_l = G.mapping(z_l, c=None, modes_idx=modes_idx_l)
    w_c = G.mapping(z_c, c=None, modes_idx=modes_idx_c)
    w_r = G.mapping(z_r, c=None, modes_idx=modes_idx_r)
    ws_context = torch.stack([w_l, w_r], dim=1)

    for frame_idx in tqdm(range(num_frames), desc=f'Generating infinite images (of length {num_frames}) for {args.exp_name}'):
        if frame_idx > 0 and frame_idx % (G.synthesis_cfg.patchwise.w_coord_dist // 2) == 0:
            z_l = z_c
            z_c = z_r
            z_r = torch.randn_like(z_l)

            modes_idx_l = modes_idx_c
            modes_idx_c = modes_idx_r
            modes_idx_r = torch.randint(0, G.synthesis_cfg.num_modes, size=(batch_size,), device=device)

            w_l = w_c
            w_c = w_r
            w_r = G.mapping(z_r, c=None, modes_idx=modes_idx_r)
            ws_context = torch.stack([w_l, w_r], dim=1)

        shift = (frame_idx % (G.synthesis_cfg.patchwise.w_coord_dist // 2)) * G.synthesis_cfg.patchwise.grid_size
        left_borders_idx = torch.zeros(batch_size, device=device).long() + shift
        imgs = G.synthesis(w_c, ws_context=ws_context, left_borders_idx=left_borders_idx, noise='const')
        imgs = imgs.cpu().clamp(-1, 1) * 0.5 + 0.5

        for line_idx, img in enumerate(imgs):
            img = TVF.to_pil_image(img)
            img.save(f'{save_dir}/{line_idx:03d}_{frame_idx:06d}.png')
