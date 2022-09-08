from argparse import Action
from genericpath import isfile
import sys

from cv2 import normalize
sys.path.append('D:/projects/eg3d')
import cv2
import click
import torch
import os
import copy
import legacy
import glob
import imageio
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, utils
import dnnlib
import json
from typing import List, Optional, Tuple, Union
import re
from torchvision.utils import save_image
from torch_utils.ops import conv2d_gradfix
from torch_utils import misc
from inversion.BiSeNet import BiSeNet
from dnnlib.seg_tools import initFaceParsing, parsing_img, face_parsing, mask2color
from training.volumetric_rendering import create_cam2world_matrix, sample_camera_positions

#----------------------------------------------------------------------------

def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List[int]]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_tuple(s: Union[str, Tuple[int,int]]) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.

    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    if m := re.match(r'^(\d+)[x,](\d+)$', s):
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')

#----------------------------------------------------------------------------

@click.command()
@click.option("--drive_root", type=str, default=None)
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--encoder', 'encoder_pkl', help='Network pickle filename', required=True)
@click.option('--grid', 'grid_dims', type=parse_tuple, help='Grid width/height, e.g. \'4x3\' (default: 1x1)', default=(1,1))
@click.option('--seeds', type=parse_range, help='List of random seeds', required=True)
@click.option('--outdir', help='Output directory', type=str, required=True, metavar='DIR')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
def run_video_animation(drive_root, network_pkl, encoder_pkl, grid_dims, seeds, outdir, truncation_psi, truncation_cutoff):
    grid_w = grid_dims[0]
    grid_h = grid_dims[1]
    mp4 = os.path.join(outdir, 'reenact.mp4')
    device = torch.device('cuda')

    print('Loading generator from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    print('Loading encoder from "%s"...' % encoder_pkl)
    with dnnlib.util.open_url(encoder_pkl) as f:
        E = legacy.load_encoder_pkl(f)['E'].to(device)

    os.makedirs(outdir, exist_ok=True)

    bisNet, to_tensor = initFaceParsing(path='pretrained_models', device=device)

    ws_avg = G.mapping.w_avg[None, None, :] 

    video_out = imageio.get_writer(mp4, mode='I', fps=30, codec='libx264', bitrate='10M')

    with open(os.path.join(drive_root, 'dataset.json'), 'rb') as f:
        label_list = json.load(f)['labels']
    label_list = dict(label_list)
    c_front = torch.tensor([1,0,0,0, 0,1,0,0, 0,0,1,2.7, 0,0,0,1, 4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).float().to(device).reshape(1, -1)

    img_list = sorted(glob.glob(drive_root+'/*.png'))
    ws = []
    for seed in seeds:
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        w = G.mapping(z, c_front, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        ws.append(w)
    ws = torch.cat(ws)
    batch = len(seeds)

    for k, img_path in tqdm(enumerate(img_list)):
        if k > 100:
            break
        target_img = np.array(Image.open(img_path))  
        target_img = torch.tensor(target_img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        target_img = target_img / 127.5 - 1.    
        target_seg = face_parsing(target_img, bisNet=bisNet) * 2. - 1.
        c = [label_list[os.path.basename(img_path)]]
        c = np.array(c)
        c[:, [1,2,5,6,9,10]] *= -1 
        c = torch.tensor(c.astype({1: np.int64, 2: np.float32}[c.ndim])).to(device)

        render_params = {
                "fov": 16,
                "num_steps": 96, 
                }

        cond_imgs = G.synthesis(ws, c.repeat(batch, 1), noise_mode='const', render_params=render_params)
        rec_ws = E(cond_imgs, target_seg.repeat(batch, 1, 1, 1))
        rec_ws =  rec_ws + ws_avg # start from average

        img = G.synthesis(rec_ws, c.repeat(batch, 1), noise_mode='const', render_params=render_params)
        result = torch.cat([target_img, img], 0)
        video_out.append_data(layout_grid(result, grid_w=grid_w, grid_h=grid_h))
    video_out.close()
            

if __name__ == "__main__":
    with torch.no_grad():
        run_video_animation()