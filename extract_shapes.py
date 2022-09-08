from cv2 import normalize
import plyfile
import argparse
import torch
import numpy as np
import skimage.measure
import scipy
import mrcfile
import os
from tqdm import tqdm
import os
import re
from typing import List, Optional, Tuple, Union
import math
import click
import dnnlib
import PIL.Image
import copy
import legacy
import torch.nn as nn
import torchvision
import cv2
from torch_utils import misc
from torchvision.utils import save_image

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

def create_samples(N=512, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3
                   
    return samples.unsqueeze(0), voxel_origin, voxel_size


def sample_generator_ide3d(generator, aux_img_net, z, c, max_batch=100000, voxel_resolution=256, voxel_origin=[0,0,0], cube_length=2.0, psi=0.5, **kwargs):
    n_regions = 3
    head = 0
    samples, voxel_origin, voxel_size = create_samples(voxel_resolution, voxel_origin, cube_length)
    samples = 0.9 * (samples).to(z.device)
    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=z.device)
    
    with torch.no_grad():
        # mapping latent codes into W space
        ws = generator.mapping(z, c, truncation_psi=psi)

        # TODO: compute nerf input
        voxel_block_ws = []
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, generator.synthesis.num_ws, generator.synthesis.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in generator.synthesis.voxel_block_resolutions:
                block = getattr(generator.synthesis, f'vb{res}')
                voxel_block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv
            for res in generator.synthesis.block_resolutions:
                block = getattr(generator.synthesis, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x_v = img_v = seg_v = None
        for res, cur_ws in zip(generator.synthesis.voxel_block_resolutions, voxel_block_ws):
            block = getattr(generator.synthesis, f'vb{res}')
            x_v, img_v, seg_v = block(x_v, img_v, cur_ws, condition_img=seg_v)

        render_kwargs = {}
        render_kwargs["img_size"] = generator.synthesis.render_size
        render_kwargs["nerf_noise"] = 0
        render_kwargs["ray_start"] = 2.25
        render_kwargs["ray_end"] = 3.3
        render_kwargs["fov"] = 18
        render_kwargs.update(kwargs)

        P = c[:, :16].reshape(-1, 4, 4)
        P = P.detach()
        render_kwargs["camera"] = P
        
        # Sequentially evaluate siren with max_batch_size to avoid OOM
        while head < samples.shape[1]:
            tail = head + max_batch
            coarse_output = generator.synthesis.renderer.sample_voxel(img_v, seg_v, samples[:, head:tail]).reshape(samples.size(0), -1, 52)
            sigmas[:, head:head+max_batch] = coarse_output[:, :, -1:]
            head += max_batch
    sigmas = sigmas.reshape((voxel_resolution, voxel_resolution, voxel_resolution)).cpu().numpy()
    return sigmas


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str)
    parser.add_argument('--seeds', type=parse_range)
    parser.add_argument('--trunc', type=float, default=1.)
    parser.add_argument('--noise-mode', default='const')
    parser.add_argument('--cube_size', type=float, default=0.3)
    parser.add_argument('--voxel_resolution', type=int, default=256)
    parser.add_argument('--outdir', type=str, default='shapes')
    opt = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with dnnlib.util.open_url(opt.network) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    aux_img_net = nn.Sequential(
      nn.Conv2d(32, 3, 1),
      nn.Tanh()
    ).to(device)

    render_params = {
         "h_stddev": 0.,
        "v_stddev": 0.,
        "num_steps": 96
        }
        
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        label = torch.tensor([1,0,0,0, 0,1,0,0, 0,0,1,2.7, 0,0,0,1, 4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1])[None].float().to(device)

    os.makedirs(opt.outdir, exist_ok=True)
    
    for seed in tqdm(opt.seeds):
        torch.manual_seed(seed)
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        voxel_grid = sample_generator_ide3d(G, aux_img_net, z, label, cube_length=opt.cube_size, voxel_resolution=opt.voxel_resolution, psi=opt.trunc, **render_params)
        
        with mrcfile.new_mmap(os.path.join(opt.outdir, f'{seed}.mrc'), overwrite=True, shape=voxel_grid.shape, mrc_mode=2) as mrc:
            mrc.data[:] = voxel_grid

        np.save(os.path.join(opt.outdir, f'{seed}.npy'), voxel_grid)
