from email.policy import default
from random import random
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
from torch.utils import data
from torchvision import transforms, utils
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from tensorboardX import SummaryWriter

import sys
sys.path.append('D:/projects/IDE-3D')
from torch_utils.ops import conv2d_gradfix
from torch_utils import misc
import dnnlib
import legacy
from inversion.networks import Encoder, HybridEncoder
from inversion.BiSeNet import BiSeNet
from training.volumetric_rendering import sample_camera_positions, create_cam2world_matrix
from inversion.criteria.lpips.lpips import LPIPS
from inversion.criteria import id_loss as IDLoss
from dnnlib.seg_tools import *

"""
同时将segmap和real image输入到encoder,
segmap 输出前八层latent code;
real image 输出后10层latent code
"""

def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def launch_training(desc, outdir):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def cam_sampler(batch, device):
    camera_points, phi, theta = sample_camera_positions(device, n=batch, r=2.7, horizontal_mean=0.5*math.pi, 
                                                        vertical_mean=0.5*math.pi, horizontal_stddev=0.3, vertical_stddev=0.155, mode='gaussian')
    c = create_cam2world_matrix(-camera_points, camera_points, device=device)
    c = c.reshape(batch, -1)
    c = torch.cat((c, torch.tensor([4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).reshape(1, -1).repeat(batch, 1).to(c)), -1)
    return c


def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

    
def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
    

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch
            

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss


class VGGLoss(nn.Module):
    def __init__(self, device, n_layers=5):
        super().__init__()
        
        feature_layers = (2, 7, 12, 21, 30)
        self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)  

        vgg = torchvision.models.vgg19(pretrained=True).features
        
        self.layers = nn.ModuleList()
        prev_layer = 0
        for next_layer in feature_layers[:n_layers]:
            layers = nn.Sequential()
            for layer in range(prev_layer, next_layer):
                layers.add_module(str(layer), vgg[layer])
            self.layers.append(layers.to(device))
            prev_layer = next_layer

        for param in self.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss().to(device)

    def forward(self, source, target):
        loss = 0
        source, target = (source + 1) / 2, (target + 1) / 2
        for layer, weight in zip(self.layers, self.weights):
            source = layer(source)
            with torch.no_grad():
                target = layer(target)
            loss += weight*self.criterion(source, target)

        return loss 


def train(rank, world_size, opts):
    data        = opts.data
    seg         = opts.seg
    outdir      = opts.outdir
    g_ckpt      = opts.g_ckpt 
    e_ckpt      = opts.e_ckpt
    max_steps   = opts.max_steps
    batch       = opts.batch // opts.num_gpus
    lr          = opts.lr
    local_rank  = opts.local_rank
    vgg         = opts.vgg
    l2          = opts.l2
    adv         = opts.adv
    lpips       = opts.lpips
    id          = opts.id
    entropy     = opts.entropy
    cycle       = opts.cycle
    truncation  = opts.truncation
    start_from_latent_avg   = opts.start_from_latent_avg  
    train_gen   = opts.train_gen
    train_real  = opts.train_real
    tensorboard = opts.tensorboard
    num_gpus    = opts.num_gpus
    port        = opts.port
    outdir      = opts.outdir
    
    np.random.seed(0)
    torch.manual_seed(0)

    setup(rank, world_size, port)
    device = torch.device(rank)

    conv2d_gradfix.enabled = True  # Improves training speed.
    
    start_iter = 0 

    # load the pre-trained model
    if os.path.isdir(g_ckpt):
        g_ckpt = sorted(glob.glob(g_ckpt + '/*.pkl'))[-1]
    print('Loading generator from "%s"...' % g_ckpt)
    with dnnlib.util.open_url(g_ckpt) as fp:
        network = legacy.load_network_pkl(fp)
        G = network['G_ema'].requires_grad_(False).to(device)
        D = network['D'].requires_grad_(False).to(device)
    if e_ckpt is not None:      
        if os.path.isdir(e_ckpt):
            e_ckpt = sorted(glob.glob(e_ckpt + '/*.pkl'))[-1]
        print('Loading encoder from "%s"...' % e_ckpt)
        with dnnlib.util.open_url(e_ckpt) as fp:
            network = legacy.load_encoder_pkl(fp)
            E = network['E'].requires_grad_(True).to(device)
            start_iter = network.get('start_iter', 0)
    else:
        E = HybridEncoder(G.img_resolution, 10, 8, G.mapping.w_dim, add_dim=0, input_img_dim=3, input_seg_dim=19).to(device)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)
    D = copy.deepcopy(D).eval().requires_grad_(False).to(device)
    bisNet, to_tensor = initFaceParsing(path='pretrained_models', device=device)
    E_optim = optim.Adam(E.parameters(), lr=lr, betas=(0.9, 0.99))
    requires_grad(E, True)

    E_ddp = DDP(E, device_ids=[rank], broadcast_buffers=False)
    E = E_ddp.module

    torch.manual_seed(rank)
    # load the dataset
    training_set_kwargs = dict(class_name='training.dataset_seg.CameraLabeledDataset', path=data, seg_path=seg, load_seg=True, use_labels=True, xflip=False, resolution=G.img_resolution)
    data_loader_kwargs  = dict(pin_memory=True, prefetch_factor=2)
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)
    training_set_sampler  = misc.InfiniteSampler(dataset=training_set, rank=local_rank, num_replicas=num_gpus, seed=0)  # for now, single GPU first.
    training_set_iterator = torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=opts.batch//opts.num_gpus, **data_loader_kwargs)
    training_set_iterator = iter(training_set_iterator)
    print('Num images: ', len(training_set))
    print('Image shape:', training_set.image_shape)
    
    pbar = range(max_steps)
    pbar = tqdm(pbar, initial=start_iter, dynamic_ncols=True, smoothing=0.01)

    loss_dict  = {}
    calc_vgg_loss       = VGGLoss(device=device)
    calc_lpips_loss     = LPIPS(net_type='alex').to(device).eval()
    calc_id_loss        = IDLoss.IDLoss().to(device).eval()
    CrossEntropyLoss    = torch.nn.CrossEntropyLoss()
    ws_avg     = G.mapping.w_avg[None, None, :]

    if rank == 0:
        if SummaryWriter and tensorboard:
            logger = SummaryWriter(logdir=f'{outdir}/logs')

    for idx in pbar:
        i = idx + start_iter
        if i > max_steps:
            print("Done!")
            break
        loss_dict = {}
        E_optim.zero_grad()  # zero-out gradients
        
#----------------------- SYNTHETIC IMAGE -----------------------#
        assert train_gen or train_real 
        if train_gen:
            z_samples = np.random.randn(batch, ws_avg.size(-1))
            z_samples = torch.from_numpy(z_samples).to(torch.float32).to(device)
            cs = torch.tensor([1,0,0,0, 0,1,0,0, 0,0,1,2.7, 0,0,0,1, 4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).float().to(device).reshape(1,-1).repeat(batch,1)

            w_samples = G.mapping(z_samples, cs)
            c_samples = cam_sampler(batch, device)
            if truncation < 1.0:
                w_samples = ws_avg + (w_samples - ws_avg) * truncation
            gen_img = G.synthesis(w_samples, c=c_samples, noise_mode='const')
            gen_seg = face_parsing(gen_img, bisNet=bisNet)
            gen_seg = gen_seg * 2. - 1.
            rec_gen_ws = E_ddp(gen_img, gen_seg)
            if start_from_latent_avg:
                rec_gen_ws = rec_gen_ws + ws_avg.repeat(rec_gen_ws.shape[0], 1, 1)
            
            # LATENT LOSS
            rec_ws_loss = F.smooth_l1_loss(rec_gen_ws, w_samples).mean()
            loss_dict['loss_ws'] = rec_ws_loss * 50.

            # L2 LOSS
            rec_gen_img = G.synthesis(rec_gen_ws, c=c_samples, noise_mode='const')
            rec_gen_l2_loss = F.mse_loss(rec_gen_img, gen_img)
            loss_dict["loss_gen_l2"] = rec_gen_l2_loss * l2

            # CROSS ENTROPY LOSS
            _, rec_gen_mask = parsing_img(bisNet, rec_gen_img, argmax=False, return_mask=False, remap=False, with_grad=True)
            _, gen_mask = parsing_img(bisNet, gen_img, argmax=True, return_mask=False, remap=False, with_grad=True)
            cross_entropy_loss = CrossEntropyLoss(rec_gen_mask, gen_mask.squeeze(1))
            loss_dict["loss_gen_entropy"] = cross_entropy_loss * entropy

            # CYCLE LOSS
            _, rec_gen_seg = parsing_img(bisNet, rec_gen_img, with_grad=True)
            rec_cycle_ws = E_ddp(gen_img, rec_gen_seg)
            rec_cycle_loss =  F.smooth_l1_loss(rec_gen_ws, rec_cycle_ws).mean()
            loss_dict["loss_cycle"] = rec_cycle_loss * cycle

    #----------------------- REAL IMAGE -----------------------#
        if train_real:
            real_img, real_seg, real_label = next(training_set_iterator)
            real_img = real_img.to(device).to(torch.float32) / 127.5 - 1.
            real_seg = real_seg.to(device).to(torch.float32) * 2. - 1.
            real_label = real_label.to(device)
            rec_real_ws = E_ddp(real_img, real_seg)
            if start_from_latent_avg:
                rec_real_ws = rec_real_ws + ws_avg   # make sure it starts from the average (?)
            rec_real_img, rec_real_img_raw = G.synthesis(ws=rec_real_ws, c=real_label, return_raw=True, noise_mode='const')
            rec_real_img_raw = F.interpolate(rec_real_img_raw, rec_real_img.shape[2:], mode="bilinear")
            # rec_pred = D(torch.cat([rec_real_img, rec_real_img_raw], 1), real_label)

            # VGG LOSS
            rec_vgg_loss = calc_vgg_loss(rec_real_img, real_img)
            loss_dict["loss_vgg"] = rec_vgg_loss * vgg

            # MSE LOSS
            rec_real_l2_loss = F.mse_loss(rec_real_img, real_img)
            loss_dict["loss_real_l2"] = rec_real_l2_loss * l2

            # # ADV LOSS
            # adv_loss = g_nonsaturating_loss(rec_pred)
            # loss_dict["loss_adv"] = adv_loss * adv

            # LPIPS LOSS
            lpips_loss = calc_lpips_loss(rec_real_img, real_img)
            loss_dict['loss_lpips'] = lpips_loss * lpips

            # ID LOSS
            id_loss, _, _ = calc_id_loss(rec_real_img, real_img, real_img)
            loss_dict["loss_id"] = id_loss * id

            # CROSS ENTROPY LOSS
            _, rec_real_mask = parsing_img(bisNet, rec_real_img, argmax=False, return_mask=False, remap=False, with_grad=True)
            _, real_mask = parsing_img(bisNet, real_img, argmax=True, return_mask=False, remap=False, with_grad=True)
            real_cross_entropy_loss = CrossEntropyLoss(rec_real_mask, real_mask.squeeze(1))
            loss_dict["loss_real_entropy"] = real_cross_entropy_loss * entropy

            # CYCLE LOSS
            _, rec_real_seg = parsing_img(bisNet, rec_real_img, with_grad=True)
            rec_real_cycle_ws = E_ddp(real_img, rec_real_seg)
            rec_real_cycle_loss =  F.smooth_l1_loss(rec_real_ws, rec_real_cycle_ws).mean()
            loss_dict["loss_real_cycle"] = rec_real_cycle_loss * cycle

        E_loss = sum([loss_dict[l] for l in loss_dict])
        E_loss.backward()
        E_optim.step()

        if rank == 0:
            desp = '\t'.join([f'{name}: {loss_dict[name].item():.4f}' for name in loss_dict])
            pbar.set_description((desp))

            if SummaryWriter and tensorboard:
                logger.add_scalar('E_loss/total', E_loss.item(), i)
                for key in loss_dict.keys():
                    logger.add_scalar(f'E_loss/{key}', loss_dict[key].item(), i)       
            
            if i % 1000 == 0:
                os.makedirs(f'{outdir}/sample', exist_ok=True)
                with torch.no_grad():
                    if train_gen:
                        # visualize reconstracted synthetic exmaples
                        if rec_gen_img is None:
                            rec_gen_img = G.synthesis(ws=rec_gen_ws, c=real_label, noise_mode='const')
                        
                        gen_vis = mask2color(gen_seg.detach())
                        fused = gen_vis.detach() * 0.2 + rec_gen_img.detach() * 0.8
                        sample = torch.cat([gen_img.detach(), gen_vis.detach(), rec_gen_img.detach(), fused.detach()])
                        utils.save_image(
                            sample,
                            f"{outdir}/sample/{str(i).zfill(6)}_gen.png",
                            nrow=int(batch),
                            normalize=True,
                            range=(-1, 1),
                        )
                    
                    if train_real:
                        # visualize reconstracted real exmaples
                        if rec_real_img is None:
                            rec_real_img = G.synthesis(ws=rec_real_ws, c=real_label, noise_mode='const')
                        real_vis = mask2color(real_seg.detach())
                        fused = real_vis.detach() * 0.2 + rec_real_img.detach() * 0.8
                        sample = torch.cat([real_img.detach(), real_vis.detach(), rec_real_img.detach(), fused.detach()])
                        utils.save_image(
                            sample,
                            f"{outdir}/sample/{str(i).zfill(6)}_real.png",
                            nrow=int(batch),
                            normalize=True,
                            range=(-1, 1),
                        )
                    
                    torch.cuda.empty_cache()

            if i % 10000 == 0:
                os.makedirs(f'{outdir}/checkpoints', exist_ok=True)
                snapshot_pkl = os.path.join(f'{outdir}/checkpoints/', f'network-snapshot-{i//1000:06d}.pkl')
                snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
                snapshot_data['E'] = E_ddp.module
                snapshot_data['start_iter'] = i
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--seg", type=str, default=None)
    parser.add_argument("--g_ckpt", type=str, default=None)
    parser.add_argument("--e_ckpt", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=1000000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--vgg", type=float, default=0.0)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--adv", type=float, default=0.0)
    parser.add_argument("--lpips", type=float, default=0.0)
    parser.add_argument("--id", type=float, default=0.0)
    parser.add_argument("--entropy", type=float, default=0.0)
    parser.add_argument("--cycle", type=float, default=0.0)
    parser.add_argument("--truncation", type=float, default=1.)
    parser.add_argument("--tensorboard", type=bool, default=True)
    parser.add_argument("--start_from_latent_avg", action="store_true")
    parser.add_argument("--train_gen", action="store_true")
    parser.add_argument("--train_real", action='store_true')
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--outdir", type=str, default='')
    parser.add_argument("--port", type=str, default='12355')

    opts = parser.parse_args()
    print("training real: ", opts.train_real)
    print("start from avg", opts.start_from_latent_avg)

    train_type = ''
    if opts.train_gen: train_type += 'gen'
    if opts.train_real: train_type += 'real'
    desc = f'gpus{opts.num_gpus:d}-batch{opts.batch:d}-{train_type}-vgg{opts.vgg:.2f}-l2{opts.l2:.2f}-adv{opts.adv:.2f}'
    outdir = launch_training(outdir=opts.outdir, desc=desc)
    opts.outdir = outdir

    mp.spawn(train, args=(opts.num_gpus, opts), nprocs=opts.num_gpus, join=True)
