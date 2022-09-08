import os
import os.path as osp
import trimesh
import mcubes
import pyrender
from pyglet import gl
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
from tqdm import tqdm
from PIL import Image
import glob
import imageio
import cv2
import click

from training.volumetric_rendering import sample_camera_positions, create_cam2world_matrix

@click.command()
@click.option('--fname', type=str, help='Network pickle filename', required=True)
@click.option('--size', type=int, help='Size of mesh', default=256, required=False)
@click.option('--sigma-threshold', type=float, help='sigma threshold during marching cube', default=10., show_default=True)
@click.option('--w-frames', type=int, help='Number of frames to interpolate between latents', default=240)
@click.option('--outdir', type=str, help='output dir of video', required=True)
def render(fname, size, sigma_threshold, w_frames, outdir):
    sigma_threshold = sigma_threshold
    num = w_frames
    voxel_grid = np.load(fname)
    voxel_grid = np.maximum(voxel_grid, 0)
    vertices, triangles = mcubes.marching_cubes(voxel_grid, sigma_threshold)
    mesh = trimesh.Trimesh(vertices/size, triangles)

    # mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    # mesh.fix_normals()
    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True, material=pyrender.MetallicRoughnessMaterial(
        alphaMode='BLEND',
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        smooth=True,
        metallicFactor=0.2,
        roughnessFactor=0.6,
    ))
    
    for i in tqdm(range(num)):
        yaw = math.pi*(0.5+0.15*math.cos(2*math.pi*i/num))
        pitch = math.pi*(0.5-0.05*math.sin(2*math.pi*i/num))
        scene = pyrender.Scene(ambient_light=[5, 5, 5], bg_color=[255, 255, 255])
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 180.0 * 18)
        direc_l = pyrender.DirectionalLight(color=np.ones(3), intensity=3)

        scene.add(mesh, pose=np.eye(4))
        camera_points, phi, theta = sample_camera_positions(device=None, n=1, r=2.7, horizontal_mean=yaw, vertical_mean=pitch, mode=None)
        c = create_cam2world_matrix(-camera_points, camera_points, device=None)
        P = c.reshape(-1,4,4).numpy()[0]
        P[:3, 3] += 0.5
        scene.add(camera, pose=P)
        scene.add(direc_l, pose=P)

        # render scene
        r = pyrender.OffscreenRenderer(512, 512)
        color, _ = r.render(scene)

        im = Image.fromarray(color)
        margin = 0
        im = im.crop((margin, margin, 512-margin, 512-margin))
        id = os.path.basename(fname).split('.')[0]
        os.makedirs(f"tmp/{id}", exist_ok=True)
        im.save(f"tmp/{id}/{i:03d}.png")

    img2video(glob.glob(f"tmp/{id}/*.png"), f"{outdir}/render.mp4")
    cmd = f'rmdir /s/q tmp\\{id}'
    os.system(cmd)
    

def img2video(img_list, mp4):
    video_out = imageio.get_writer(mp4, mode='I', fps=60, codec='libx264', bitrate='10M')
    for img in img_list:
        frame = cv2.imread(img)
        video_out.append_data(frame)
    video_out.close()


if __name__ == '__main__':
    fname = 'out/1.npy'
    render()
