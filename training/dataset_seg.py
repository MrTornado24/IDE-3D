# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def seg_channels(self):
        return self.seg_shapes

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                       # Path to directory or zip.
        seg_path        = None,     # Path of segmentation data.
        load_seg        = False,    # If load segmentation maps. 
        id_remap        = False,    # It re-split segmentation classes.
        resolution      = None,     # Ensure specific resolution, None = highest available.
        non_rebalance   = True,
        **super_kwargs,             # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None
        self._seg_path = seg_path
        self.load_seg = load_seg
        self.id_remap = id_remap
        if not id_remap:
            self.color_map = {
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
        else:
            self.color_map = {
                0: [0, 0, 0],
                1: [204, 0, 0],
                2: [51, 51, 255], 
                3: [0, 0, 204], 
                4: [0, 204, 0]}
        
        # 0 background: 0
        # 1 complexion: 1, 2, 8, 9, 17
        # 2 eyes & mouth: 4, 5, 6, 7, 10, 11, 12
        # 3 hair: 13
        # 4 wearing: 3, 14, 15, 16, 18
        
        # self.remap_list = torch.tensor([0, 1, 6, 3, 3, 3, 3, 3, 6, 6, 4, 5, 5, 2, 8, 8, 8, 7, 8]).float()
        # self.remap_list_np = np.array([0, 1, 6, 3, 3, 3, 3, 3, 6, 6, 4, 5, 5, 2, 8, 8, 8, 7, 8]).astype('float')

        self.remap_list = torch.tensor([0, 1, 1, 4, 2, 2, 2, 2, 1, 1, 2, 2, 2, 3, 4, 4, 4, 1, 4])
        self.remap_list_np = np.array([0, 1, 1, 4, 2, 2, 2, 2, 1, 1, 2, 2, 2, 3, 4, 4, 4, 1, 4]).astype('float')
        self.seg_shapes = 19 if not id_remap else 5

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if non_rebalance:
            self._image_fnames = [fname for fname in self._image_fnames if int(fname[-12:-4]) < 140000]
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')
        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0, resolution).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')

        if self._seg_path is not None and self.load_seg:
            if os.path.isdir(self._seg_path):
                self._type = 'dir'
                self._all_segnames = {os.path.relpath(os.path.join(root, fname), start=self._seg_path) for root, _dirs, files in os.walk(self._seg_path) for fname in files}
            elif self._file_ext(self._seg_path) == '.zip':
                self._type = 'zip'
                self._all_segnames = set(self._get_zipfile().namelist())
            else:
                raise IOError('Path must point to a directory or zip')
            
            self._seg_fnames = sorted(fname for fname in self._all_segnames if self._file_ext(fname) in PIL.Image.EXTENSION)
            if non_rebalance:
                self._seg_fnames = [fname for fname in self._seg_fnames if int(fname[-12:-4]) < 140000]
            assert len(self._seg_fnames) == len(self._image_fnames)

        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def _open_segfile(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._seg_path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx, resolution=None):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            # if pyspng is not None and self._file_ext(fname) == '.png':
            #     image = pyspng.load(f.read())
            # else:
            image = PIL.Image.open(f)
            if resolution:
                image = image.resize((resolution, resolution))
            image = np.array(image)
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels[:, [1,2,5,6,9,10]] *= -1 # opencv --> opengl
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

    def _load_invalid_list(self):
        fname = 'invalids.txt'
        with open(os.path.join(self._path, fname), 'r') as f:
            lines = f.readlines()
        lines = [line.split('\n')[0].replace('\\', '/') for line in lines]
        return lines

    def _mask_labels(self, mask_np):
        label_size = len(self.color_map.keys())
        labels = np.zeros((label_size, mask_np.shape[0], mask_np.shape[1]))
        for i in range(label_size):
            labels[i][mask_np==i] = 1.0
        return labels

    def id_remap(self, seg):
        return self.remap_list[seg.long()]

    def id_remap_np(self, seg):
        return self.remap_list_np[seg.astype('int')]

    def mask2color(self, masks, use_argmax=True):
        if use_argmax:
            masks = torch.argmax(masks, dim=1).float()
        sample_mask = torch.zeros((masks.shape[0], masks.shape[1], masks.shape[2], 3), dtype=torch.float, device=masks.device)
        for key in self.color_map:
            sample_mask[masks==key] = torch.tensor(self.color_map[key], dtype=torch.float, device=masks.device)
        sample_mask = sample_mask.permute(0,3,1,2)
        return sample_mask

    def mask2color_np(self, masks, use_argmax=True):
        if use_argmax:
            masks = np.argmax(masks, axis=1)
        sample_mask = np.zeros((masks.shape[0], masks.shape[1], masks.shape[2], 3))
        for key in self.color_map:
            sample_mask[masks==key] = self.color_map[key]    
        sample_mask = sample_mask.transpose(0,3,1,2)
        return sample_mask

    def _load_raw_seg(self, raw_idx, resolution=None):
        fname = self._seg_fnames[raw_idx]
        with self._open_segfile(fname) as f:
            # if pyspng is not None and self._file_ext(fname) == '.png':
            #     image = pyspng.load(f.read())
            # else:
            image = PIL.Image.open(f).convert('L')
            if resolution:
                image = image.resize((resolution, resolution), resample=PIL.Image.NEAREST)
            image = np.array(image)
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

#----------------------------------------------------------------------------

class CameraLabeledDataset(ImageFolderDataset):
    
    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx], resolution=self.resolution)
        label = self.get_label(idx)
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
            seg = seg[:, :, ::-1]
            if self._use_labels:
                assert label.shape == (25,)
                label[[1,2,3,4,8]] *= -1

        if self.load_seg:
            seg = self._load_raw_seg(self._raw_idx[idx], resolution=self.resolution)
            if self.id_remap:
                seg = self.id_remap_np(seg)
            seg = self._mask_labels(seg[0])
            return image.copy(), seg.copy(), label
        else:
            return image.copy(), label