from ctypes import wstring_at
import pickle
import functools
import torch
import math
import numpy as np
from torch_utils import misc
from configs import paths_config, global_config


def toogle_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def load_tuned_G(run_id, type):
    new_G_path = f'{paths_config.checkpoints_dir}/model_{run_id}_{type}.pt'
    with open(new_G_path, 'rb') as f:
        new_G = torch.load(f).to(global_config.device).eval()
    new_G = new_G.float()
    toogle_grad(new_G, False)
    return new_G


def load_old_G():
    with open(paths_config.ide3d_ffhq, 'rb') as f:
    # with open(paths_config.stylegan2_ada_ffhq, 'rb') as f:
        old_G = pickle.load(f)['G_ema'].to(global_config.device).eval()
        old_G = old_G.float()
    return old_G


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


def sample_generator_ide3d(generator, ws, c, max_batch=100000, voxel_resolution=512, voxel_origin=[0,0,0], cube_length=0.3, psi=0.7, **kwargs):
    head = 0
    samples, voxel_origin, voxel_size = create_samples(voxel_resolution, voxel_origin, cube_length)
    samples = 0.9 * (samples).to(c.device)
    # debug:
    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=c.device)
    
    with torch.no_grad():
        # mapping latent codes into W space
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
            # coarse_output = generator.synthesis.nerf_forward(samples[:, head:tail], w[:, :14]).reshape(samples.shape[0], -1, 33) # 33
            coarse_output = generator.synthesis.renderer.sample_voxel(img_v, seg_v, samples[:, head:tail]).reshape(samples.size(0), -1, 52)
            sigmas[:, head:head+max_batch] = coarse_output[:, :, -1:]
            head += max_batch
    sigmas = sigmas.reshape((voxel_resolution, voxel_resolution, voxel_resolution)).cpu().numpy()
    return sigmas


def smooth_ws(ws):
    ws_p = ws[2:-2] + 0.75 * ws[3:-1] + 0.75 * ws[1:-3] + 0.25 * ws[:-4] + 0.25 * ws[4:]
    ws_p = ws_p / 3
    return ws_p
