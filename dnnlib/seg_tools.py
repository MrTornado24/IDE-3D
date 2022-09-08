import os
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms, utils

import sys
sys.path.append('D:/projects/eg3d')
from inversion.BiSeNet import BiSeNet


COLOR_MAP = {
            0: [0, 0, 0], 
            1: [204, 0, 0],
            2: [76, 153, 0], 
            3: [204, 204, 0], 
            4: [51, 51, 255], 
            5: [204, 0, 204], 
            6: [0, 255, 255], 
            7: [255, 204, 204], 
            8: [102, 51, 0], 
            9: [255, 0, 0], 
            10: [102, 204, 0], 
            11: [255, 255, 0], 
            12: [0, 0, 153], 
            13: [0, 0, 204], 
            14: [255, 51, 153], 
            15: [0, 204, 204], 
            16: [0, 51, 0], 
            17: [255, 153, 51], 
            18: [0, 204, 0]}


label_list = {
            'background': 0,
            'skin': 1, 
            'nose': 2, 
            'eye_g': 3, 
            'l_eye': 4,
            'r_eye': 5,
            'l_brow': 6, 
            'r_brow': 7, 
            'l_ear': 8,
            'r_ear': 9, 
            'mouth': 10,
            'u_lip': 11,
            'l_lip': 12, 
            'hair': 13, 
            'hat': 14, 
            'ear_r': 15, 
            'neck_l': 16, 
            'neck': 17, 
            'cloth': 18
      }


color_list = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]
remap_list = torch.tensor([0, 1, 6, 7, 4, 5, 2, 2, 10, 11, 12, 8, 9, 15, 3, 17, 16, 18, 13, 14]).float()



def id_remap(seg, type='sof'):
    return remap_list[seg.long()].to(seg.device)


def mask2label_np(mask_np):
    label_size = len(COLOR_MAP.keys())
    labels = np.zeros((label_size, mask_np.shape[0], mask_np.shape[1]))
    for i in range(label_size):
        labels[i][mask_np==i] = 1.0
    return labels
            

def mask2color(masks):
    masks = torch.argmax(masks, dim=1).float()
    sample_mask = torch.zeros((masks.shape[0], masks.shape[1], masks.shape[2], 3), dtype=torch.float, device=masks.device)
    for key in COLOR_MAP:
        sample_mask[masks==key] = torch.tensor(COLOR_MAP[key], dtype=torch.float, device=masks.device)
    sample_mask = sample_mask.permute(0,3,1,2)
    return sample_mask


def mask2color_np(masks):
    masks = np.argmax(masks, axis=1)
    sample_mask = np.zeros((masks.shape[0], masks.shape[1], masks.shape[2], 3))
    for key in COLOR_MAP:
        sample_mask[masks==key] = COLOR_MAP[key]    
    sample_mask = sample_mask.transpose(0,3,1,2)
    return sample_mask


def scatter(condition_img, classSeg=19, label_size=(512, 512)):
    batch, c, height, width = condition_img.size()
    if height != label_size[0] or width != label_size[1]:
        condition_img= F.interpolate(condition_img, size=label_size, mode='nearest')
    input_label = torch.zeros(batch, classSeg, *label_size, device=condition_img.device)
    return input_label.scatter_(1, condition_img.long(), 1)


def parsing_img(bisNet, img, argmax=True, return_mask=True, with_grad=False, remap=True):
    if not with_grad:
        with torch.no_grad():
            segmap = bisNet(img)[0]
            if argmax:
                segmap = segmap.argmax(1, keepdim=True)
            if remap:
                segmap = id_remap(segmap, 'celebahq')
    else:
        segmap = bisNet(img)[0]
        if argmax:
            segmap = segmap.argmax(1, keepdim=True)
        if remap:
            segmap = id_remap(segmap, 'celebahq')
    if return_mask:
            segmap = scatter(segmap)
    return img, segmap


def face_parsing(img, bisNet):
    img = F.interpolate(img, size=(512, 512), mode='bilinear', align_corners=True)
    _, seg_label = parsing_img(bisNet, img)
    return seg_label


def initFaceParsing(n_classes=20, path=None, device='cuda:0'):
    net = BiSeNet(n_classes=n_classes).to(device)
    net.load_state_dict(torch.load(path+'/segNet-20Class.pth'))
    net.eval()
    to_tensor = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

    ])
    return net, to_tensor


def label2color_np(mask, save_path):
      im_base = np.zeros((512, 512, 3))
      for idx, color in enumerate(color_list):
            im_base[mask == idx] = color
      result = Image.fromarray((im_base).astype(np.uint8))
      result.save(save_path)


def mask2color_custom(masks, attr_list):
    masks = torch.argmax(masks, dim=1).float()
    sample_mask = torch.zeros((masks.shape[0], masks.shape[1], masks.shape[2], 3), dtype=torch.float, device=masks.device)
    for key in COLOR_MAP:
        if key in attr_list:
            sample_mask[masks==key] = torch.tensor(COLOR_MAP[key], dtype=torch.float, device=masks.device)
        else:
            sample_mask[masks==key] = torch.tensor((255,255,255), dtype=torch.float, device=masks.device)
    sample_mask = sample_mask.permute(0,3,1,2)
    return sample_mask

    
def label2mask(path, attr_list=None):
    target_seg = Image.open(path).convert('L')
    target_seg = np.array(target_seg)
    target_seg_flip = target_seg[:, ::-1].copy()
    target_seg = mask2label_np(target_seg)
    target_seg = torch.tensor(target_seg).unsqueeze(0).float() * 2 - 1
    target_seg_flip = mask2label_np(target_seg_flip)
    target_seg_flip = torch.tensor(target_seg_flip).unsqueeze(0).float() * 2 - 1
    utils.save_image(mask2color(target_seg), f'{path.split(".")[0]}_color.png', normalize=True)
    if attr_list is not None:
        utils.save_image(mask2color_custom(target_seg, attr_list), f'{path}_color_attr.png', normalize=True)


def switch_semantic(attributes, ref_mask_path, tar_mask_path, offset_x=0, offset_y=0, mask_root = '', maskcolor_root = ''):
      ref_im = Image.open(ref_mask_path)
      ref_im = np.array(ref_im)
      tar_im = Image.open(tar_mask_path)
      tar_im = np.array(tar_im)
      im_temp = tar_im.copy()
      for attribute in attributes:
            im_temp[tar_im == label_list[attribute]] = 1
      # debug:
      (x_hair, y_hair) = np.where(im_temp==13)
      for attribute in attributes:
            (x,y) = np.where(ref_im==label_list[attribute])
            x += offset_x # down: "x+"; up: "x-";
            y += offset_y
            x[x>511] = 511
            y[y>511] = 511
            im_temp[(x, y)] = label_list[attribute]
            im_temp[(x_hair, y_hair)] = 13
      result = Image.fromarray((im_temp).astype(np.uint8))
      tar_index = os.path.basename(tar_mask_path).split('.')[0]
      ref_index = os.path.basename(ref_mask_path).split('.')[0]
      save_name = f'mask_{tar_index}_switch_{ref_index}_{"_".join(attributes)}.png'
      result.save(os.path.join(mask_root, save_name))
      save_name = f'maskcolor_{tar_index}_switch_{ref_index}_{"_".join(attributes)}.png'
      label2color_np(im_temp, os.path.join(maskcolor_root, save_name))


def rm_semantic(attributes, mask_path, mask_root='', maskcolor_root=''):
      im = Image.open(mask_path)
      im = np.array(im)
      im_temp = im.copy()
      for attribute in attributes:
            im_temp[im == label_list[attribute]] = 1
      result = Image.fromarray((im_temp).astype(np.uint8))
      index = os.path.basename(mask_path).split('.')[0]
      save_name = f'mask_{index}_no_{"_".join(attributes)}.png'
      result.save(os.path.join(mask_root, save_name))
      save_name = f'maskcolor_{index}_no_{"_".join(attributes)}.png'
      label2color_np(im_temp, os.path.join(maskcolor_root, save_name))


if __name__ == '__main__':
    mask_root = 'data/ffhq/masks512x512'
    maskcolor_root = 'data/ffhq/maskcolors512x512'
    attributes = ['mouth', 'u_lip', 'l_lip']  
    # attributes = ['hair']
    # attributes = ['l_brow', 'r_brow']   
    # attributes = ['eye_g'] 
    
    target = '0819110515_segmap.png'
    source = 'data/ffhq/masks512x512/00000/img00000458.png'

    switch_semantic(attributes, source, target, offset_x=-12, offset_y=0) # x:+下;y:+右
    # rm_semantic(attributes, target)

    # label2mask('0828162048_segmap.png', [13])


    # D:/projects/sofgan/ui_reic_results/00239_orig/0812231002_segmap.png