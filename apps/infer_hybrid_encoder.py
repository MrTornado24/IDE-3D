from audioop import cross
from email.policy import default
from genericpath import isfile
from random import random
import sys
sys.path.append('D:/projects/IDE-3D')
import os
import numpy as np
import torch
import copy
import torch.distributed as dist
import torchvision
import click
import pickle
import glob
import math
import re
import json
from PIL import Image
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
from torch_utils.ops import conv2d_gradfix
from torch_utils import misc
from torchvision import transforms, utils
from tqdm import tqdm
from tensorboardX import SummaryWriter
import dnnlib
import legacy
from inversion.networks import Encoder, HybridEncoder
from inversion.BiSeNet import BiSeNet
from training.volumetric_rendering import sample_camera_positions, create_cam2world_matrix
from inversion.criteria.lpips.lpips import LPIPS
from inversion.criteria import id_loss as IDLoss
from dnnlib.seg_tools import *

    
def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
    
            
def test(target_img, target_seg, target_code, target_label, outdir, G, E, new_G, bisNet, start_from_latent_avg, device):

    random_seed = 0
    np.random.seed(random_seed)

    conv2d_gradfix.enabled = True  # Improves training speed.
    
    ws_avg = G.mapping.w_avg[None, None, :]
    
    img_id = os.path.basename(target_img).split('.')[0]
    target_img = Image.open(target_img)
    target_img = np.array(target_img)
    target_img = torch.tensor(target_img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    target_img = target_img / 127.5 - 1.

    if target_seg is not None:
        target_seg = Image.open(target_seg).convert('L')
        target_seg = np.array(target_seg)
        target_seg = mask2label_np(target_seg)
        target_seg = torch.tensor(target_seg).unsqueeze(0).float().to(device) * 2 - 1
    else:
        target_seg = face_parsing(target_img, bisNet=bisNet) * 2. - 1.
    ### visulize target image and target segmap
    # utils.save_image(torch.cat((target_img, mask2color(target_seg.clone())), -1),
    #                 f"debug.png",
    #                 nrow=int(1),
    #                 normalize=True,
    #                 range=(-1, 1))

    ### save semantic mask
    # _, rec_seg = parsing_img(bisNet, target_img, remap=True, return_mask=False)
    # rec_seg = rec_seg[0, 0].detach().cpu().numpy().astype(np.uint8)
    # rec_seg = Image.fromarray(rec_seg, mode='L')
    # rec_seg.save(f'cele_segmap.png')


    rec_ws = E(target_img, target_seg)
    if start_from_latent_avg:
        rec_ws += ws_avg.repeat(*rec_ws.shape[:2], 1)

    if target_code is not None:
        target_code = torch.load(target_code).to(device)
        rec_ws = torch.cat((rec_ws[:, :8], target_code[:, 8:]), 1)

    if new_G is not None:
        rec_img = new_G.synthesis(ws=rec_ws, c=target_label, noise_mode='const')
    else:
        rec_img = G.synthesis(ws=rec_ws, c=target_label, noise_mode='const')

    os.makedirs(f"{outdir}/{img_id}", exist_ok=True)

    torch.save(rec_ws, f'{outdir}/{img_id}/rec_ws.pt')
    ### visulize rec image and target segmap
    utils.save_image(rec_img,
                    f"{outdir}/{img_id}/rec_img.png",
                    nrow=int(1),
                    normalize=True,
                    range=(-1, 1))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--target_img", type=str, default='img.png')
    parser.add_argument("--target_seg", type=str, default=None)
    parser.add_argument("--target_code", type=str, default=None)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--g_ckpt", type=str, default=None)
    parser.add_argument("--e_ckpt", type=str, default=None)
    parser.add_argument("--new_g_ckpt", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--start_from_latent_avg", type=bool, default=True)
    parser.add_argument("--outdir", type=str, default='')
    opts = parser.parse_args()

    ### load generators and encoders
    device = torch.device('cuda', 0)
    print('Loading generator from "%s"...' % opts.g_ckpt)
    try:
        with open(opts.new_g_ckpt, 'rb') as f:
            new_G = torch.load(f).to(device).eval()
        new_G = new_G.float()
    except:
        new_G = None
    
    with dnnlib.util.open_url(opts.g_ckpt) as fp:
        network = legacy.load_network_pkl(fp)
        G = network['G_ema'].requires_grad_(False).to(device)
    print('Loading encoder from "%s"...' % opts.e_ckpt)
    with dnnlib.util.open_url(opts.e_ckpt) as fp:
        network = legacy.load_encoder_pkl(fp)
        E = network['E'].requires_grad_(False).to(device)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)
    E = copy.deepcopy(E).eval().requires_grad_(False).to(device)
    bisNet, to_tensor = initFaceParsing(path='pretrained_models')

    fname = 'D:/projects/eg3d/data/FFHQ/images512x512/dataset.json' # label list Path. 
    with open(fname, 'rb') as f:
        label_list = json.load(f)['labels']
    label_list = dict(label_list)

    if os.path.isfile(opts.target_img):
        c = [label_list[opts.target_img[-21:]]]
        c = np.array(c)
        c[:, [1,2,5,6,9,10]] *= -1 
        c = torch.tensor(c.astype({1: np.int64, 2: np.float32}[c.ndim])).to(device)
        test(
                target_img=opts.target_img, 
                target_seg=opts.target_seg,
                target_code=opts.target_code, 
                target_label=c,
                outdir=opts.outdir, 
                G=G,
                E=E, 
                new_G=new_G, 
                bisNet=bisNet, 
                start_from_latent_avg=opts.start_from_latent_avg, 
                device=device)

    elif os.path.isdir(opts.target_img):
        img_path = sorted(glob.glob(opts.target_img + '/*.png'))
        for img_name in img_path:
            img_id = int(img_name[-12:-4])
            img_name = f'data/ffhq/images512x512/{(img_id//1000):05d}/img{img_id:08d}.png' # Path/to/Your/Image/Folders
            seg_name = f'data/ffhq/masks512x512/{(img_id//1000):05d}/img{img_id:08d}.png'  # Path/to/Your/Segmap/Folders
            
            c = [label_list[img_name[-21:]]]
            c = np.array(c)
            c[:, [1,2,5,6,9,10]] *= -1 
            c = torch.tensor(c.astype({1: np.int64, 2: np.float32}[c.ndim])).to(device)
            test(
                target_img=img_name, 
                target_seg=None,
                target_code=opts.target_code, 
                target_label=c,
                outdir=opts.outdir, 
                G=G,
                E=E, 
                new_G=new_G, 
                bisNet=bisNet, 
                start_from_latent_avg=opts.start_from_latent_avg, 
                device=device)
