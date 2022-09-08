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
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, utils
import dnnlib
import json
from torch_utils.ops import conv2d_gradfix
from torch_utils import misc
from inversion.BiSeNet import BiSeNet
from dnnlib.seg_tools import initFaceParsing, parsing_img, face_parsing, mask2color
from training.volumetric_rendering import create_cam2world_matrix, sample_camera_positions


@click.command()
@click.option("--target", type=str, default=None)
@click.option("--g_ckpt", type=str, default=None)
@click.option("--e_ckpt", type=str, default=None)
@click.option("--batch", type=int, default=8)
@click.option("--seed", type=int, default=22)
@click.option("--local_rank", type=int, default=0)
@click.option("--outdir", type=str, required=True)
@click.option("--save_video", default=True)
@click.option("--trunc", default=1.)
@click.option("--start_from_latent_avg", type=bool, default=True)
def run_video_style_transfer(target, outdir, g_ckpt, e_ckpt, batch, seed, local_rank, save_video, trunc, start_from_latent_avg):
    random_seed = seed
    np.random.seed(random_seed)
    conv2d_gradfix.enabled = True  # Improves training speed.
    device = torch.device('cuda', local_rank)
    os.makedirs(outdir, exist_ok=True)

    with dnnlib.util.open_url(g_ckpt) as fp:
        network = legacy.load_network_pkl(fp)
        G = network['G_ema'].requires_grad_(False).to(device)
    print('Loading encoder from "%s"...' % e_ckpt)
    with dnnlib.util.open_url(e_ckpt) as fp:
        network = legacy.load_encoder_pkl(fp)
        E = network['E'].requires_grad_(False).to(device)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)
    E = copy.deepcopy(E).eval().requires_grad_(False).to(device)
    bisNet, to_tensor = initFaceParsing(path='pretrained_models', device=device)

    ws_avg = G.mapping.w_avg[None, None, :] 

    save_path = f'{outdir}/restyle.mp4'
    nrows,ncols = 1, 2
    width_pad, height_pad = 2 * (ncols + 1), 2 * (nrows + 1) 
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(save_path, fourcc, 20, (512 * ncols + width_pad, 512 * nrows + height_pad))

    ### prepare label list
    fname = 'D:/projects/eg3d-pose-detection-main/obama_datasets.json'
    with open(fname, 'rb') as f:
        label_list = json.load(f)['labels']
    label_list = dict(label_list)
    c_front = torch.tensor([1,0,0,0, 0,1,0,0, 0,0,1,2.7, 0,0,0,1, 4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).float().to(device).reshape(1, -1)

    img_list = sorted(glob.glob(target+'/*.png'))
    for k, img_path in tqdm(enumerate(img_list)):
        if k > 70:
            break
        img_render = Image.open(img_path)
        target_img = img_render.crop((0,0,512,512))  
        target_img = np.array(target_img)  
        target_img = torch.tensor(target_img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        target_img = target_img / 127.5 - 1.    
        target_render = img_render.crop((512,0,2*512, 512))
        target_render = np.array(target_render)  
        target_render = torch.tensor(target_render.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        target_render = target_render / 127.5 - 1.
        c = [label_list[f'{int(os.path.basename(img_path).split(".")[0]):05d}.png']]
        c = np.array(c)
        c[:, [1,2,5,6,9,10]] *= -1 
        c = torch.tensor(c.astype({1: np.int64, 2: np.float32}[c.ndim])).to(device)

        # style mixing
        render_params = {
                "fov": 16,
                "num_steps": 96, 
            }
        batch = nrows * ncols - 1
        ### debug: select great z samples!
        # for i in range(100):
        #     z_samples = np.random.RandomState(i).randn(1, ws_avg.size(-1))
        #     z_samples = torch.from_numpy(z_samples).to(torch.float32).to(device)
        #     w_samples = G.mapping(z_samples, c_front.repeat(1, 1), truncation_psi=trunc)
        #     gen_imgs = G.synthesis(w_samples, c.repeat(1, 1), noise_mode='const', render_params=render_params)
        #     utils.save_image(gen_imgs, f"tmp/seed_{i}.png", normalize=True, range=(-1, 1))
        # seeds = [14, 15, 44, 39, 89]
        seeds = [44]
        for seed in seeds:
            z_samples = np.random.RandomState(seed).randn(1, ws_avg.size(-1))
            z_samples = torch.from_numpy(z_samples).to(torch.float32).to(device)
            w_samples = G.mapping(z_samples, c_front, truncation_psi=trunc)
            
        # result = [target_img.detach().cpu()]
        result = [target_render.detach().cpu()]
        rec_ws = w_samples
        rec_gen_imgs = G.synthesis(rec_ws, c.repeat(batch, 1), cond_img=target_render, noise_mode='const', render_params=render_params)
        
        for j in range(batch):
            result.append(rec_gen_imgs[j][None].detach().cpu())

        # # debug
        # result.append((mask2color(target_seg) / 255. * 2. - 1.).detach().cpu())

        result = torch.cat(result, dim=0)
        result = (utils.make_grid(result, nrow=2) + 1) / 2 * 255
        result = result.numpy()[::-1].transpose((1, 2, 0)).astype('uint8')
        out.write(result)
        result = []

        if k == 360:
            trajectory = np.linspace(0,1,20)
            for t in trajectory:
                result = [target_img.detach().cpu()]
                c_t = c * (1-t) + c_front * (t)
                rec_gen_imgs = G.synthesis(rec_ws, c_t.repeat(batch, 1), noise_mode='const',render_params=render_params)
                for j in range(batch):
                    result.append(rec_gen_imgs[j][None].detach().cpu())

                result = torch.cat(result, dim=0)
                result = (utils.make_grid(result, nrow=3) + 1) / 2 * 255
                result = result.numpy()[::-1].transpose((1, 2, 0)).astype('uint8')
                out.write(result)
                result = []
            trajectory = set_trajectory('orbit')
            
            for yaw, pitch in trajectory:
                result = [target_img.detach().cpu()]
                render_params = {
                "h_mean": yaw,
                "v_mean": pitch,
                "h_stddev": 0.,
                "v_stddev": 0.,
                "fov": 16,
                "num_steps": 96, 
                }
                camera_points, phi, theta = sample_camera_positions(device, n=1, r=2.7, horizontal_mean=yaw, vertical_mean=pitch, mode=None)
                c_t = create_cam2world_matrix(-camera_points, camera_points, device=device)
                c_t = c_t.reshape(1,-1)
                c_t = torch.cat((c_t, torch.tensor([4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).reshape(1, -1).to(c_t)), -1)
                rec_gen_imgs = G.synthesis(rec_ws, c_t.repeat(batch, 1), noise_mode='const',render_params=render_params)
                for j in range(batch):
                    result.append(rec_gen_imgs[j][None].detach().cpu())

                result = torch.cat(result, dim=0)
                result = (utils.make_grid(result, nrow=3) + 1) / 2 * 255
                result = result.numpy()[::-1].transpose((1, 2, 0)).astype('uint8')
                out.write(result)
                result = []
            trajectory = np.linspace(0,1,20)
            render_params = {
                "fov": 16,
                "num_steps": 96, 
                }
            for t in trajectory:
                result = [target_img.detach().cpu()]
                c_tmp = c_t * (1-t) + c * (t)
                rec_gen_imgs = G.synthesis(rec_ws, c_tmp.repeat(batch, 1), noise_mode='const',render_params=render_params)
                for j in range(batch):
                    result.append(rec_gen_imgs[j][None].detach().cpu())

                result = torch.cat(result, dim=0)
                result = (utils.make_grid(result, nrow=3) + 1) / 2 * 255
                result = result.numpy()[::-1].transpose((1, 2, 0)).astype('uint8')
                out.write(result)
                result = []



            
    out.release()

import math
def set_trajectory(traj_type):
    traj = []
    images = []
    if traj_type == 'front':
        for frame_idx in range(240):
            h = math.pi*(0.5+0.1*math.cos(2*math.pi*frame_idx/(0.5 * 240)))
            v = math.pi*(0.5-0.05*math.sin(2*math.pi*frame_idx/(0.5 * 240)))
            traj.append((h, v))
    elif traj_type == 'orbit':
        for t in np.linspace(0.5, 0.3, 15):
            pitch = math.pi/2
            yaw = t * math.pi
            traj.append((yaw, pitch))
        for t in np.linspace(0.3, 0.5, 15):
            pitch = math.pi/2
            yaw = t * math.pi
            traj.append((yaw, pitch))
        for t in np.linspace(0.5, 0.7, 15):
            pitch = math.pi/2
            yaw = t * math.pi
            traj.append((yaw, pitch))
        for t in np.linspace(0.7, 0.5, 15):
            pitch = math.pi/2
            yaw = t * math.pi
            traj.append((yaw, pitch))
        for t in np.linspace(0.5, 0.4, 15):
            yaw = math.pi/2
            pitch = t * math.pi
            traj.append((yaw, pitch))
        for t in np.linspace(0.4, 0.5, 15):
            yaw = math.pi/2
            pitch = t * math.pi
            traj.append((yaw, pitch))
        for t in np.linspace(0.5, 0.6, 15):
            yaw = math.pi/2
            pitch = t * math.pi
            traj.append((yaw, pitch))
        for t in np.linspace(0.6, 0.5, 15):
            yaw = math.pi/2
            pitch = t * math.pi
            traj.append((yaw, pitch))
        
    return traj

if __name__ == "__main__":
    with torch.no_grad():
        run_video_style_transfer() # pylint: disable=no-value-for-parameter