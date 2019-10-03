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

CLASS_MAPPING = {'12': 0, '123': 1, '13': 2, '2': 3, '23': 4, '24': 5, '3': 6, '4': 7, '5': 8}

class TiffFolder(Dataset):
    def __init__(self, 
                data_dir,
                class_mapping=CLASS_MAPPING,
                transform=None):
        super(TiffFolder, self).__init__()
        
        self._transform      = transform
        if self._transform is None:
            self._transform = transforms.Compose([transforms.ToTensor(),])

        self._image_paths    = []
        self._labels         = []
        self._label_to_index = class_mapping
        self._index_to_label = dict()

        for label in os.listdir(data_dir):
            for image_path in glob.glob(os.path.join(data_dir, label, "*.tif")):
                self._image_paths.append(image_path)
                self._labels.append(self._label_to_index[label])

    def label_to_index(self, label):
        return self._label_to_index[label]

    def index_to_label(self, class_index):
        return self._index_to_label[class_index]

    def __getitem__(self, index):
        image = tiff.imread(self._image_paths[index])
        # replace no_value with mean
        h, w, c = image.shape
        for i in range(c):
            image[..., i][image[..., i] == NODATA_VALUE] = DATASET_MEAN[i]
        
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

        if x >= self.patch_size and y >= self.patch_size and x < self.image.shape[1] - self.patch_size and y < self.image.shape[0] - self.patch_size:
            left_x      = x - self.patch_size
            right_x     = x + self.patch_size + 1

            bot_y       = y - self.patch_size
            top_y       = y + self.patch_size + 1
            
            patch       = self.image[bot_y: top_y, left_x: right_x, :]
            indices_y, indices_x, indices_c = np.where(patch < NODATA_VALUE)
            for id_y, id_x, id_c in zip(indices_y, indices_x, indices_c):
                image[id_y, id_x, id_c] = DATASET_MEAN[c]
            
            return y, x, self._transform(patch)
        else:
            patch   = np.zeros((2 * self.patch_size + 1, 2 * self.patch_size + 1, self.image.shape[2]), dtype=np.float32)
            return -1, -1, self._transform(patch)

    def __len__(self):
        return self.image.shape[0] * self.image.shape[1]