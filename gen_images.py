"""Generate images using pretrained network pickle."""
import math
import os
import re
from typing import List, Optional, Tuple, Union
from torchvision.utils import save_image
import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from training.volumetric_rendering import sample_camera_positions, create_cam2world_matrix
import legacy
from dnnlib.seg_tools import *

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

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
):
    """Generate multi-viewed images and semantic masks using pretrained network pickle.
    """
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Generate images.
    cs = torch.tensor([1,0,0,0, 0,1,0,0, 0,0,1,2.7, 0,0,0,1, 4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).float().to(device).reshape(1,-1)
    for seed_idx, seed in enumerate(seeds):
        torch.manual_seed(seed)
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        ws = G.mapping(z=z, c=cs, truncation_psi=truncation_psi)
        yaws = [-0.5, 0, 0.5]
        imgs, segs = [], []
        for k, yaw in enumerate(yaws):
            render_params = {
                "h_mean": yaw+math.pi*0.5,
                "v_mean": math.pi * 0.5,
                "h_stddev": 0.,
                "v_stddev": 0.,
                "fov": 18,
                "num_steps": 96, 
            }
            camera_points, phi, theta = sample_camera_positions(device, n=1, r=2.7, horizontal_mean=yaw+math.pi*0.5, vertical_mean=math.pi * 0.5, mode=None)
            c = create_cam2world_matrix(-camera_points, camera_points, device=device)
            c = c.reshape(1,-1)
            c = torch.cat((c, torch.tensor([4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).reshape(1, -1).to(c)), -1)

            img, seg = G.synthesis(ws, c=c, render_params=render_params, noise_mode=noise_mode, return_seg=True)
            seg = (mask2color(seg) / 255. - 0.5) / 0.5 # (0, 255) -> (-1, 1) 
            imgs.append(img)
            segs.append(seg)
        imgs = torch.cat(imgs)
        segs = torch.cat(segs)
        save_image(imgs, f'{outdir}/seed{seed:04d}.png', normalize=True, range=(-1, 1))
        save_image(segs, f'{outdir}/seed{seed:04d}_seg.png', normalize=True, range=(-1, 1))


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
