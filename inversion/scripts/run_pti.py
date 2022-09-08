from ctypes import util
from email.policy import default
from random import choice, randrange
from re import L
from string import ascii_uppercase
from cv2 import normalize, resize
from matplotlib import image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import os
import sys
import wandb
import torch
import imageio
from PIL import Image
import numpy as np
import math
import mrcfile
import torch.nn.functional as F
from tqdm import tqdm
import argparse

import torchvision.utils as utils

sys.path.append('D:/projects/IDE-3D/inversion')
from configs import global_config, paths_config
from training.coaches.multi_id_coach import MultiIDCoach
from training.coaches.single_id_coach import SingleIDCoach
from utils.ImagesDataset import ImagesDataset
from utils.models_utils import load_old_G, load_tuned_G, sample_generator_ide3d, sample_generator_ide3d
from training.volumetric_rendering import sample_camera_positions, create_cam2world_matrix



def display_alongside_source_image(images): 
    res = np.concatenate([np.array(image) for image in images], axis=1) 
    return Image.fromarray(res) 


def plot_syn_images(syn_images, image_name):
    # syn_images = torch.cat(syn_images, -1)
    syn_images = syn_images[-1]
    img = (syn_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()[0] 
    resized_image = Image.fromarray(img,mode='RGB')
    os.makedirs(f'{paths_config.experiments_output_dir}/{run_name}/{image_name}', exist_ok=True)
    resized_image.save(f'{paths_config.experiments_output_dir}/{run_name}/{image_name}/rec_{image_name}.png')


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


def run_PTI(run_name='', use_wandb=False, use_multi_id_training=False, image_name=None, label_path=None, projector_type='ide3d', initial_w=None):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = global_config.cuda_visible_devices

    if run_name == '':
        global_config.run_name = ''.join(choice(ascii_uppercase) for i in range(12))
    else:
        global_config.run_name = run_name

    if use_wandb:
        run = wandb.init(project=paths_config.pti_results_keyword, reinit=True, name=global_config.run_name)
    global_config.pivotal_training_steps = 1
    global_config.training_step = 1

    embedding_dir_path = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}/{paths_config.pti_results_keyword}'
    os.makedirs(embedding_dir_path, exist_ok=True)

    dataset = ImagesDataset(paths_config.input_data_path, transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        resolution=(512, 512))

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    if use_multi_id_training:
        coach = MultiIDCoach(dataloader, use_wandb, image_name=image_name, projector_type=projector_type)
    else:
        coach = SingleIDCoach(dataloader, use_wandb, image_name=image_name, label_path=label_path, projector_type=projector_type, initial_w=initial_w)

    coach.train()

    return global_config.run_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--image_name', type=str, default=None)
    parser.add_argument('--label_path', type=str, default=None)
    parser.add_argument('--projector_type', type=str, default='ide3d')
    parser.add_argument('--initial_w', type=str, default=None)
    parser.add_argument('--pivotal_tuning', action='store_true')
    parser.add_argument('--viz_image', action='store_true')
    parser.add_argument('--viz_video', action='store_true')
    parser.add_argument('--viz_mesh', action='store_true')
    args = parser.parse_args()

    if args.run_name is None:
        assert args.pivotal_tuning
    
    if args.pivotal_tuning:
        run_name = run_PTI(run_name=args.run_name if args.run_name is not None else '', use_wandb=False, use_multi_id_training=False, 
                            image_name=args.image_name, label_path=args.label_path, projector_type=args.projector_type, initial_w=args.initial_w)
    else:
        run_name = args.run_name

    import glob
    flag = 0
    image_path = sorted(glob.glob(f'{paths_config.input_data_path}/*.png'))
    for image_name in image_path:
        image_name = os.path.basename(image_name).split('.')[0]
        if args.image_name is not None:
            if image_name != args.image_name:
                flag = 1
                continue
        old_G = load_old_G()
        new_G = load_tuned_G(run_name, type=image_name)
        w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
        embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
        w_pivot = torch.load(f'{embedding_dir}/0.pt')
        c = torch.load(f'{embedding_dir}/0_label.pt')
        
        if args.viz_image:
            ### visualization
            if args.projector_type.startswith('ide3d'):
                old_image = old_G.synthesis(w_pivot, c=c, noise_mode='const', force_fp32 = True)
                new_image = new_G.synthesis(w_pivot, c=c, noise_mode='const', force_fp32 = True)
            else:
                old_image = old_G.synthesis(w_pivot, noise_mode='const', force_fp32 = True)
                new_image = new_G.synthesis(w_pivot, noise_mode='const', force_fp32 = True)
            print('Upper image is the inversion before Pivotal Tuning and the lower image is the product of pivotal tuning')
            plot_syn_images([old_image, new_image], image_name)

        if args.viz_video:
            ### free-view rendering
            device = w_pivot.device
            video_out = imageio.get_writer(f'{paths_config.experiments_output_dir}/{run_name}/{image_name}/rec_{image_name}.mp4', mode='I', fps=60, codec='libx264', bitrate='12M')
            os.makedirs(f'{paths_config.experiments_output_dir}/{run_name}/{image_name}/frames', exist_ok=True)
            for frame_idx in tqdm(range(240)):
                h = math.pi*(0.5+0.1*math.cos(2*math.pi*frame_idx/(0.5 * 240)))
                v = math.pi*(0.5-0.05*math.sin(2*math.pi*frame_idx/(0.5 * 240)))
                render_params = {
                    "h_mean": h,
                    "v_mean": v,
                    "h_stddev": 0.,
                    "v_stddev": 0.,
                    "fov": 18,
                    "num_steps": 96
                }
                camera_points, phi, theta = sample_camera_positions(device=device, n=1, r=2.7, horizontal_mean=h, vertical_mean=v, mode=None)
                c = create_cam2world_matrix(-camera_points, camera_points, device=device)
                c = c.reshape(1, -1)
                c = torch.cat((c, torch.tensor([4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).reshape(1, -1).repeat(1, 1).to(c)), -1)

                img, img_raw = new_G.synthesis(ws=w_pivot, c=c, render_params=render_params, return_raw=True, noise_mode='const')
                img_raw = F.interpolate(img_raw, img.shape[2:], mode="bilinear")

                ### debug ###
                if frame_idx % 1  == 0:
                    utils.save_image(
                        img, 
                        f'{paths_config.experiments_output_dir}/{run_name}/{image_name}/frames/{frame_idx:05d}.png',
                        normalize=True,
                        range=(-1,1)
                    )

                video_out.append_data(layout_grid(img, grid_w=1, grid_h=1))
                # video_out.append_data(layout_grid(torch.stack([img_raw[0], img[0]]), grid_w=2, grid_h=1))
            video_out.close()

        if args.viz_mesh:
            ### extract shapes
            render_params = {
                "h_stddev": 0.,
                "v_stddev": 0.,
                "num_steps": 96
            }
            voxel_grid = sample_generator_ide3d(new_G, ws=w_pivot, c=c, cube_length=1.0, voxel_resolution=256, **render_params)
            with mrcfile.new_mmap(f'{paths_config.experiments_output_dir}/{run_name}/{image_name}/rec_mesh.mrc', overwrite=True, shape=voxel_grid.shape, mrc_mode=2) as mrc:
                    mrc.data[:] = voxel_grid
            
        if flag:
            break