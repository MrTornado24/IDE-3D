from email.policy import default
from random import random
import sys
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
import argparse
from PIL import Image
from torch import nn, autograd, optim
from torch.nn import functional as F
from torchvision import transforms, utils
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from typing import List, Optional, Tuple, Union
from tensorboardX import SummaryWriter

sys.path.append('D:/projects/IDE-3D')
import dnnlib
import legacy
from torch.utils import data
from torch_utils.ops import conv2d_gradfix
from torch_utils import misc
from inversion.networks import Encoder, HybridEncoder, MultiViewHybridEncoder
from inversion.BiSeNet import BiSeNet
from training.volumetric_rendering import sample_camera_positions, create_cam2world_matrix
from inversion.criteria.lpips.lpips import LPIPS
from inversion.criteria import id_loss as IDLoss
from train_hybrid_encoder import initFaceParsing, face_parsing, parsing_img, requires_grad, mask2color, mask2label_np



def parse_range(s: Union[str, List]) -> List[int]:
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


@click.command()
@click.option("--target_img", type=str, default=None)
@click.option("--target_seg", type=str, default=None)
@click.option("--target_code", type=str, default=None)
@click.option("--target_label", type=str, default=None)
@click.option("--g_ckpt", type=str, default=None)
@click.option("--e_ckpt", type=str, default=None)
@click.option("--local_rank", type=int, default=0)
@click.option("--start_from_latent_avg", type=bool, default=True)
@click.option("--outdir", type=str, default='')
@click.option("--max-steps", type=int, default=1000)
def finetune(target_img, target_seg, target_code, target_label, outdir, g_ckpt, e_ckpt, local_rank, start_from_latent_avg, max_steps):
    random_seed = 0
    np.random.seed(random_seed)

    conv2d_gradfix.enabled = True  # Improves training speed.
    device = torch.device('cuda', local_rank)
    
    # load the pre-trained model
    if g_ckpt.endswith('.pt'):
        print('Loading generator from "%s"...' % g_ckpt)
        with open(g_ckpt, 'rb') as f:
            G = torch.load(f).to(device).eval()
        G = G.float()
    else:
        with dnnlib.util.open_url(g_ckpt) as fp:
            network = legacy.load_network_pkl(fp)
            G = network['G_ema'].requires_grad_(False).to(device)
    if e_ckpt is not None:      
        if os.path.isdir(e_ckpt):
            e_ckpt = sorted(glob.glob(e_ckpt + '/*.pkl'))[-1]
        print('Loading encoder from "%s"...' % e_ckpt)
        with dnnlib.util.open_url(e_ckpt) as fp:
            network = legacy.load_encoder_pkl(fp)
            E = network['E'].requires_grad_(True).to(device)
    else:
        raise NotImplementedError("FAIL TO FIND PRETRAINED MODEL!")

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)
    bisNet, to_tensor = initFaceParsing(path='pretrained_models', device=device)
    opt_param = list(E.convs_seg.parameters()) + list(E.projector_seg.parameters())
    E_optim = optim.Adam(opt_param, lr=0.00001, betas=(0.9, 0.99))
    requires_grad(E, True)
    ws_avg = G.mapping.w_avg[None, None, :]
    idx = os.path.basename(target_img).split('.')[0]

    os.makedirs(f"{outdir}/{idx}", exist_ok=True)

    if target_label is not None:
        c = torch.load(target_label)
    else:
        # prepare label list
        fname = 'D:/projects/eg3d/data/ffhq/images512x512/dataset.json'
        with open(fname, 'rb') as f:
            label_list = json.load(f)['labels']
        label_list = dict(label_list)
        c = [label_list[f'{idx[3:8]}/'+idx+'.png']]
        c = np.array(c)
        c[:, [1,2,5,6,9,10]] *= -1 
        c = torch.tensor(c.astype({1: np.int64, 2: np.float32}[c.ndim])).to(device)
    c_target = c
        
    target_code = torch.load(target_code).to(device)
    # debug
    rec_gen_img = G.synthesis(target_code, c=c, noise_mode='const')
    # with torch.no_grad():
    #     target_img = G.synthesis(ws=target_code, c=c, noise_mode='const')
    #     target_seg = face_parsing(target_img, bisNet=bisNet) * 2. - 1.
    #     ### save semantic mask
    #     _, rec_seg = parsing_img(bisNet, target_img, remap=True, return_mask=False)
    #     rec_seg = rec_seg[0, 0].detach().cpu().numpy().astype(np.uint8)
    #     rec_seg = Image.fromarray(rec_seg, mode='L')
    #     rec_seg.save(f'{outdir}/{idx}/cele_segmap.png')
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

    ### Start finetune encoder
    lambda_ws = 10.
    lambda_l2 = 1.
    lambda_entropy = 1.
    lambda_cycle = 1.
    pbar = range(max_steps)
    pbar = tqdm(pbar, initial=0, dynamic_ncols=True, smoothing=0.01)
    loss_dict  = {}
    for k in pbar:
        if k > max_steps:
            print("Done!")
            break
        loss_dict = {}
        E_optim.zero_grad()  # zero-out gradients
        rec_gen_ws = E(target_img, target_seg)
        if start_from_latent_avg:
            rec_gen_ws = rec_gen_ws + ws_avg.repeat(rec_gen_ws.shape[0], 1, 1)
        # debug:
        rec_gen_ws[:, 8:] = target_code[:, 8:].detach().clone()

        rec_ws_loss = F.smooth_l1_loss(rec_gen_ws, target_code).mean()
        loss_dict['loss_ws'] = rec_ws_loss * lambda_ws

        rec_gen_img = G.synthesis(rec_gen_ws, c=c, noise_mode='const')
        rec_gen_l2_loss = F.mse_loss(rec_gen_img, target_img)
        loss_dict["loss_gen_l2"] = rec_gen_l2_loss * lambda_l2

        # CROSS ENTROPY LOSS
        _, rec_gen_mask = parsing_img(bisNet, rec_gen_img, argmax=False, return_mask=False, remap=False, with_grad=True)
        _, gen_mask = parsing_img(bisNet, target_img, argmax=True, return_mask=False, remap=False, with_grad=True)
        entropy_loss = torch.nn.CrossEntropyLoss()(rec_gen_mask, gen_mask.squeeze(1))
        loss_dict["loss_real_entropy"] = entropy_loss * lambda_entropy

        # CYCLE LOSS
        _, rec_gen_seg = parsing_img(bisNet, rec_gen_img, with_grad=True)
        rec_real_cycle_ws = E(target_img, rec_gen_seg)
        rec_real_cycle_loss =  F.smooth_l1_loss(rec_gen_ws, rec_real_cycle_ws).mean()
        loss_dict["loss_real_cycle"] = rec_real_cycle_loss * lambda_cycle

        E_loss = sum([loss_dict[l] for l in loss_dict])
        E_loss.backward()
        E_optim.step()

        desp = '\t'.join([f'{name}: {loss_dict[name].item():.4f}' for name in loss_dict])
        pbar.set_description((desp))
        
        os.makedirs(f'{outdir}/frames', exist_ok=True)
    
    # save finetuned model
    os.makedirs(f"{outdir}/{idx}/checkpoints", exist_ok=True)
    snapshot_pkl = os.path.basename(e_ckpt).split('.')[0]
    snapshot_pkl = os.path.join(f"{outdir}/{idx}/checkpoints", f'{snapshot_pkl}_tuned_{idx}.pkl')
    snapshot_data = dict()
    snapshot_data['E'] = E
    with open(snapshot_pkl, 'wb') as f:
        pickle.dump(snapshot_data, f)

    ### free-view rendering
    rec_gen_ws = E(target_img, target_seg)
    rec_gen_ws = rec_gen_ws + ws_avg.repeat(rec_gen_ws.shape[0], 1, 1)
    rec_gen_ws[:, 8:] = target_code[:, 8:].detach().clone()

    render_params = {
            "fov": 18,
            "num_steps": 96
        }
    img = G.synthesis(ws=rec_gen_ws, c=c_target, render_params=render_params, noise_mode='const')
    _, rec_seg = parsing_img(bisNet, img, remap=True, return_mask=False)
    rec_seg = rec_seg[0, 0].detach().cpu().numpy().astype(np.uint8)
    rec_seg = Image.fromarray(rec_seg, mode='L')
    rec_seg.save(f'{outdir}/{idx}/mask.png')


if __name__ == "__main__":
    finetune()