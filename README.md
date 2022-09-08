# IDE-3D: Interactive Disentangled Editing for High-Resolution 3D-aware Portrait Synthesis 

## SIGGRAPH ASIA 2022 (Journal Track)

![Teaser image](./docs/rep.png)

[![arXiv](https://img.shields.io/badge/arXiv-2108.00946-b31b1b.svg)](https://arxiv.org/abs/2205.15517)

**IDE-3D: Interactive Disentangled Editing for High-Resolution 3D-aware Portrait Synthesis**<br>
[Jingxiang Sun](https://mrtornado24.github.io/), [Xuan Wang](https://xuanwangvc.github.io/), [Yichun Shi](https://seasonsh.github.io/), [Lizhen Wang](https://lizhenwangt.github.io/), [Jue Wang](https://juewang725.github.io/), [Yebin Liu](http://www.liuyebin.com/)<br><br>
<br>https://mrtornado24.github.io/IDE-3D/<br>

Abstract: *Existing 3D-aware facial generation methods face a dilemma in quality versus editability: they either generate editable results in low resolution, or high quality ones with no editing flexibility. In this work, we propose a new approach that brings the best of both worlds together. Our system consists of three major components: (1) a 3D-semantics-aware generative model that produces view-consistent, disentangled face images and semantic masks; (2) a hybrid GAN inversion approach that initialize the latent codes from the semantic and texture encoder, and further optimized them for faithful reconstruction; and (3) a canonical editor that enables efficient manipulation of semantic masks in canonical view and producs high quality editing results. Our approach is competent for many applications, e.g. free-view face drawing, editing and style control. Both quantitative and qualitative results show that our method reaches the state-of-the-art in terms of photorealism, faithfulness and efficiency.*


## Installation

- ```git clone --recursive https://github.com/MrTornado24/IDE-3D.git ```
- ```cd IDE-3D```
- ```conda env create -f environment.yml```

## Getting started

Please download our pre-trained checkpoints from [link](https://drive.google.com/drive/folders/1-i1WLR5YCXOKjNuQEB6ECf-JxMyvqC4l?usp=sharing) and put them under `pretrained_models/`. The link mainly contains the pretrained generator `ide3d-ffhq-64-512.pkl` and the style encoder `encoder-base-hybrid.pkl`. More pretrianed models will be released soon. 


## Semantic-aware image synthesis

```.bash
# Generate videos using pre-trained model

python gen_videos.py --outdir=out --trunc=0.7 --seeds=0-3 --grid=2x2 \
    --network=pretrained_models/ide3d-ffhq-64-512.pkl --interpolate 1 --image_mode image_seg

# Generate the same 4 seeds in an interpolation sequence

python gen_videos.py --outdir=out --trunc=0.7 --seeds=0-3 --grid=1x1 \
    --network=pretrained_models/ide3d-ffhq-64-512.pkl --interpolate 1 --image_mode image_seg
```

```.bash
# Generate images using pre-trained model

python gen_images.py --outdir=out --trunc=0.7 --seeds=0-3 \
    --network=pretrained_models/ide3d-ffhq-64-512.pkl
```

```.bash
# Extract shapes (saved as .mrc and .npy) using pre-trained model

python extract_shapes.py --outdir out --trunc 0.7 --seeds 0-3 \
    --network networks/network_snapshot.pkl --cube_size 1 
    
# Render meshes to video

python render_mesh.py --fname out/0.npy --outdir out
```

We visualize our .mrc shape files with [UCSF Chimerax](https://www.cgl.ucsf.edu/chimerax/). Please refer to [EG3D](https://github.com/NVlabs/eg3d) for detailed instruction of Chimerax.


## Interactive editing

![UI](./docs/ui.gif)

We provide an interactive tool that can be used for 3D-aware face drawing and editng in real-time. Before using it, please install the enviroment with `pip install -r ./Painter/requirements.txt`.

```.bash
python Painter/run_ui.py
    --g_ckpt pretrained_models/ide3d-ffhq-64-512.pkl 
    --e_ckpt pretrained_models/encoder-base-hybrid.pkl
```

## Preparing datasets

**FFHQ**: Download and process the [Flickr-Faces-HQ dataset](https://github.com/NVlabs/ffhq-dataset) following [EG3D](https://github.com/NVlabs/eg3d). Then, parse semantic masks for all processed images using a [pretrained parsing model](https://drive.google.com/file/d/17H1JR-UJllJ3TCnEbtJscx_GgupTBtqS/view?usp=sharing). The processed data would be placed as:

```
    ├── /path/to/dataset
    │   ├── masks512x512
    │   ├── maskscolor512x512
    │   ├── images512x512
    │   │   ├── 00000
                ├──img00000000.png
    │   │   ├── ...
    │   │   ├── dataset.json

```

**Custom dataset**: You can process your own dataset using the following commands. It would be useful for real portrait image editing.

```
cd dataset_preprocessing/ffhq
python preprocess_in_the_wild.py --indir=INPUT_IMAGE_FOLDER
```

## Real portrait image editing

IDE-3D supports 3D-aware real protrait image editing using our interactive tool. Please run the following commands: 

```.bash
# infer latent code as initialization

python apps/infer_hybrid_encoder.py 
    --target_img /path/to/img_0.png
    --g_ckpt pretrained_models/ide3d-ffhq-64-512.pkl 
    --e_ckpt pretrained_models/encoder-base-hybrid.pkl
    --outdir out
```
The above command would return `rec_ws.pt` under `out/img_0`.

```.bash
# run pti

python inversion/scripts/run_pti.py 
    --run_name ide3d_plus_initial_code 
    --projector_type ide3d_plus 
    --pivotal_tuning
    --viz_image 
    --viz_mesh 
    --viz_video 
    --label_path /path/to/dataset.json 
    --image_name img_0
    --initial_w out/img_0/rec_ws.pt

```
We adopt [PTI](https://github.com/danielroich/PTI) for 3D inverison. Before running, please place the images into `examples/`. You can pass Flag `ide3d_plus` or `ide3d` to choose different inversion types ('w' and 'w+'). Flag `initial_w` specifies the latent code obtained from the last step. It benefits more reasonable shape especially for images with steep viewing angles. The command would return pose label `label.pt`, reconstructed latent code `latent.pt`, finetuned generator and some visualizations.

```.bash
# (optional) finetune encoder

python apps/finetune_hybrid_encoder.py
    --target_img /path/to/img_0.png
    --target_code /path/to/latent.pt
    --target_label /path/to/label.pt 
    --g_ckpt /path/to/finetuned_generator.pt 
    --e_ckpt pretrained_models/encoder-base-hybrid.pkl 
    --outdir out 
    --max-steps 1000

```
This step is to align the shapes reconstructed by encoders and PTI. The finetuned encoder would be saved as `finetuned_encoder.pkl`. Besides, a semantic mask `mask.png` would be saved under the same folder.  

```.bash
# run UI

python Painter/run_ui.py
    --g_ckpt /path/to/finetuned_generator.pt
    --e_ckpt /path/to/finetuned_encoder.pkl
    --target_code /path/to/latent.pt
    --target_label /path/to/label.pt
    --inversion

```
**Note** you should click `Open Image` and load `mask.png` that is returned in the last step. 

## 3D-aware CLIP-guided domain adaptation

Please obtain the adapted generators following [IDE3D-NADA](https://github.com/MrTornado24/ide3d-nada/blob/main/README.md). You can perform interactive editing in other domains by simply replacing the original generator by the adapted one:

```.bash
python Painter/run_ui.py
    --g_ckpt /path/to/adapted_generator.pt
    --e_ckpt pretrained_models/encoder-base-hybrid.pkl
```

## Semantic-guided style animation

![Teaser image](./docs/animation.png)

IDE-3D supports animating stylized virtual faces through semantic masks. Please process a video clip and prepare a `dataset.json`. Then run:

```.bash

python apps/infer_face_animation.py 
    --drive_root /path/to/images
    --network pretrained_models/ide3d-ffhq-64-512.pkl 
    --encoder pretrained_models/encoder-base-hybrid.pkl
    --grid 4x1 
    --seeds 52,197,229
    --outdir out
```

## Training

Training scipts will be released soon. 

## Acknowledgments

Part of the codes are borrowed from [StyleGAN3](https://github.com/NVlabs/stylegan3), [PTI](https://github.com/danielroich/PTI), [EG3D](https://github.com/NVlabs/eg3d) and [StyleGAN-nada](https://github.com/rinongal/StyleGAN-nada).


## Citation

If you use this code for your research, please cite the following works:
```
@article{sun2022ide,
  title={IDE-3D: Interactive Disentangled Editing for High-Resolution 3D-aware Portrait Synthesis},
  author={Sun, Jingxiang and Wang, Xuan and Shi, Yichun and Wang, Lizhen and Wang, Jue and Liu, Yebin},
  journal={arXiv preprint arXiv:2205.15517},
  year={2022}
}

@inproceedings{sun2022fenerf,
  title={Fenerf: Face editing in neural radiance fields},
  author={Sun, Jingxiang and Wang, Xuan and Zhang, Yong and Li, Xiaoyu and Zhang, Qi and Liu, Yebin and Wang, Jue},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7672--7682},
  year={2022}
}
```
