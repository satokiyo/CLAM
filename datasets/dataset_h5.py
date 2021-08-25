from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle

from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils, models
import torch.nn.functional as F

from PIL import Image
import h5py
from utils.file_utils import open_hdf5_file

from random import randrange

def eval_transforms(pretrained=False):
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    else:
        mean = (0.5,0.5,0.5)
        std = (0.5,0.5,0.5)

    trnsfrms_val = transforms.Compose(
                    [
                     transforms.ToTensor(),
                     transforms.Normalize(mean = mean, std = std)
                    ]
                )

    return trnsfrms_val

class Whole_Slide_Bag(Dataset):
    def __init__(self,
        file_path,
        pretrained=False,
        custom_transforms=None,
        target_patch_size=-1,
        ):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            pretrained (bool): Use ImageNet transforms
            custom_transforms (callable, optional): Optional transform to be applied on a sample
        """
        self.pretrained=pretrained
        if target_patch_size > 0:
            self.target_patch_size = (target_patch_size, target_patch_size)
        else:
            self.target_patch_size = None

        if not custom_transforms:
            self.roi_transforms = eval_transforms(pretrained=pretrained)
        else:
            self.roi_transforms = custom_transforms

        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            dset = f['imgs']
            self.length = len(dset)

        self.summary()
            
    def __len__(self):
        return self.length

    def summary(self):
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file['imgs']
        for name, value in dset.attrs.items():
            print(name, value)

        print('pretrained:', self.pretrained)
        print('transformations:', self.roi_transforms)
        if self.target_patch_size is not None:
            print('target_size: ', self.target_patch_size)

    def __getitem__(self, idx):
        with h5py.File(self.file_path,'r') as hdf5_file:
            img = hdf5_file['imgs'][idx]
            coord = hdf5_file['coords'][idx]
        
        img = Image.fromarray(img)
        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size)
        img = self.roi_transforms(img).unsqueeze(0)
        return img, coord

class Whole_Slide_Bag_FP(Dataset):
    def __init__(self,
        file_path,
        wsi,
        pretrained=False,
        custom_transforms=None,
        custom_downsample=1,
        target_patch_size=-1,
        target=None,
        detection_loc=False,
        skip_flag=False,
        ):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            pretrained (bool): Use ImageNet transforms
            custom_transforms (callable, optional): Optional transform to be applied on a sample
            custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
            target_patch_size (int): Custom defined image size before embedding
            target (str): detection or segmentation
            detection_loc (bool): Return detection_loc dataset additionally. detection_loc dataset must exists. i.e. detection forward must be completed.
            skip_flag (bool): Whether thresholds have been changed or not. If True(=all unchanged), skip
        Returns:

        """
        self.pretrained=pretrained
        self.wsi = wsi
        self.target = target
        self.file_path = file_path
        self.skip_flag = skip_flag
        if not custom_transforms:
            self.roi_transforms = eval_transforms(pretrained=pretrained)
        else:
            self.roi_transforms = custom_transforms

        #with h5py.File(self.file_path, "r") as f:
        with open_hdf5_file(self.file_path, mode="r") as f:
            dset = f[f'/{target}']
            self.patch_level = dset.attrs.get('patch_level')
            self.patch_size = dset.attrs.get('patch_size')
            self.slide_id = f.attrs.get('name')
            if target_patch_size > 0:
                self.target_patch_size = (target_patch_size, ) * 2
            elif custom_downsample > 1:
                self.target_patch_size = (self.patch_size // custom_downsample, ) * 2
            else:
                self.target_patch_size = None

            coords_patches = []
            names_parent_patches = []
            def get_dataset_coord(name, obj):
                '''
                パッチ座標の読み込み。以下の階層にdatasetがある。
                /target/contourxx/coords_patches
                '''
                #if isinstance(obj, h5py.Dataset) and (name.split("/")[-1] == 'coord'):
                if isinstance(obj, h5py.Dataset) and (name.split("/")[-1] == 'coords_patches'):
                    #print(obj.name)
                    #print(obj.parent.name)
                    #coords_patches.append((obj.parent.name, obj[:]))
                    coords_patches.extend(obj[:])
                    names_parent_patches.extend(obj.parent.name for _ in range(obj[:].shape[0]))
            for cont in dset:
                dset[cont].visititems(get_dataset_coord)
            self.coords_patches = np.vstack(coords_patches)
            self.names_parent_patches = names_parent_patches
            self.length = len(coords_patches)

            if detection_loc:
                coords_detection_loc_x = []
                coords_detection_loc_y = []
                names_detection_loc_patches = []
                def get_dataset_detection_loc(name, obj):
                    '''
                    核検出座標の読み込み。以下の階層にdatasetがある。
                    /target/contourxx/detection_loc_x
                    /target/contourxx/detection_loc_y
                    '''
                    if isinstance(obj, h5py.Dataset) and (name.split("/")[-1] == 'detection_loc_x'):
                        #print(obj.name)
                        #print(obj.parent.name)
                        #coords_detection_loc.append((obj.parent.name, obj[:].tolist()))
                        coords_detection_loc_x.extend(obj[:].tolist())
                        names_detection_loc_patches.extend(obj.parent.name for _ in range(obj[:].shape[0]))
                    if isinstance(obj, h5py.Dataset) and (name.split("/")[-1] == 'detection_loc_y'):
                        #print(obj.name)
                        #print(obj.parent.name)
                        #coords_detection_loc.append((obj.parent.name, obj[:].tolist()))
                        coords_detection_loc_y.extend(obj[:].tolist())
 
                dset.visititems(get_dataset_detection_loc)
                self.detection_loc_x_patches = coords_detection_loc_x
                self.detection_loc_y_patches = coords_detection_loc_y
                self.names_detection_loc_patches = names_detection_loc_patches
            else:
                self.detection_loc_x_patches = None
                self.detection_loc_y_patches = None
                self.names_detection_loc_patches = None
            f.flush()
            f.close()

        self.summary()
            
    def __len__(self):
        return self.length

    def summary(self):
        #hdf5_file = h5py.File(self.file_path, "r")
        hdf5_file = open_hdf5_file(self.file_path, mode="r")
        dset = hdf5_file[f'/{self.target}']
        for name, value in dset.attrs.items():
            print(name, value)

        print('\nfeature extraction settings')
        print('target patch size: ', self.target_patch_size)
        print('pretrained: ', self.pretrained)
        print('transformations: ', self.roi_transforms)
        hdf5_file.close()

    def __getitem__(self, idx):
        grp_name_parent, coord = self.names_parent_patches[idx], self.coords_patches[idx]
        if self.detection_loc_x_patches and self.detection_loc_y_patches:
            assert len(self.names_detection_loc_patches) == len(self.detection_loc_x_patches) == len(self.detection_loc_y_patches)
            _grp_name_parent, detection_loc_x, detection_loc_y = self.names_detection_loc_patches[idx], self.detection_loc_x_patches[idx], self.detection_loc_y_patches[idx]
            detection_loc = np.array([(x,y) for x, y in zip(detection_loc_x, detection_loc_y)])
            assert grp_name_parent == _grp_name_parent # datasets coord and detection_loc is under the same directory hierarchy.
            # 閾値が一つも変更されていない場合、かつ既に結果があるパッチは飛ばす
            if self.skip_flag:
                with h5py.File(self.file_path, "r") as f:
                #f = open_hdf5_file(self.file_path, mode="r")
                    if ('detection_dab_intensity' in f[grp_name_parent]) and ('detection_tc_positive_indices' in f[grp_name_parent]):
                        f.flush()
                        f.close()
                        return 0

            img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
    
            if self.target_patch_size is not None:
                img = img.resize(self.target_patch_size)
            img = self.roi_transforms(img)#.unsqueeze(0)

            return img, coord, grp_name_parent, detection_loc

        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size)
        img = self.roi_transforms(img)#.unsqueeze(0)
 
        return img, coord, grp_name_parent




class Dataset_All_Bags(Dataset):

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df['slide_id'][idx]




