import os
import glob

import numpy as np
import random

from PIL import Image
import tifffile as tiff

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as TF

from params import * 

class TiffFolder(Dataset):
    def __init__(self, 
                data_dir, 
                transform=None):
        super(TiffFolder, self).__init__()
        
        self._transform      = transform
        if self._transform is None:
            self._transform = transforms.Compose([transforms.ToTensor(),])

        self._image_paths    = []
        self._labels         = []
        self._label_to_index = dict()
        self._index_to_label = dict()

        for index, label in enumerate(os.listdir(data_dir)):
            self._label_to_index[label]  = index
            self._index_to_label[index]  = label
            
            for image_path in glob.glob(os.path.join(data_dir, label, "*.tif")):
                self._image_paths.append(image_path)
                self._labels.append(index)

    def label_to_index(self, label):
        return self._label_to_index[label]

    def index_to_label(self, class_index):
        return self._index_to_label[class_index]

    def __getitem__(self, index):
        image = tiff.imread(self._image_paths[index])
        
        return self._transform(image), self._labels[index]

    def __len__(self):
        return len(self._image_paths)


class TiffImageSet(Dataset):
    """docstring for TiffImage"""
    def __init__(self, image_path, transform=None, patch_size=4):
        super(TiffImageSet, self).__init__()
        
        self._transform      = transform
        if self._transform is None:
            self._transform = transforms.Compose([transforms.ToTensor(),])

        self.image          = tiff.imread(image_path)
        self.patch_size     = patch_size

    def get_shape(self):
        return self.image.shape[0], self.image.shape[1]

    def __getitem__(self, index):
        y, x        = index // self.image.shape[1], index % self.image.shape[1]
        if y - self.patch_size >= 0 and y + self.patch_size  < self.image.shape[0] \
            and x - self.patch_size >= 0 and x + self.patch_size < self.image.shape[1]:
            img_patch   = self.image[y - self.patch_size: y + self.patch_size + 1, x - self.patch_size: x + self.patch_size + 1, :]
            return y, x, self._transform(img_patch)
        else:
            img_patch   = np.zeros((2 * self.patch_size + 1, 2 * self.patch_size + 1, self.image.shape[2]), dtype=np.float32)
            return -1, -1, self._transform(img_patch)

    def __len__(self):
        return self.image.shape[0] * self.image.shape[1]