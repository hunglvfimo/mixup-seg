import os
import glob

import numpy as np

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
                transform=None,
                mixup=False,
                stage="train"):
        super(TiffFolder, self).__init__()
        
        self._transform      = transform
        if self._transform is None:
            self._transform = transforms.Compose([transforms.ToTensor(),])

        self.stage           = stage

        self._image_paths    = []
        self._labels         = []
        self._weights        = []

        self._label_to_index = dict()
        self._index_to_label = dict()

        for label_name in os.listdir(data_dir):
            if mixup:
                # split mixed label to list of single labels
                label = [c for c in label_name]
                # add single label to label dict
                for c in label:
                    if c not in self._label_to_index.keys():
                        self._label_to_index[c] = len(self._label_to_index.keys())
                        self._index_to_label[len(self._label_to_index.keys())] = c
                # 
                for image_path in glob.glob(os.path.join(data_dir, label_name, "*.tif")):
                    for c in label:
                        self._image_paths.append(image_path)
                        self._labels.append(self._label_to_index[c])
                        self._weights.append(double(1.0 / len(label)))
            else:
                if label_name not in self._label_to_index.keys():
                    self._label_to_index[label_name] = len(self._label_to_index.keys())
                    self._index_to_label[len(self._label_to_index.keys())] = label_name

                for image_path in glob.glob(os.path.join(data_dir, label_name, "*.tif")):
                    self._image_paths.append(image_path)
                    self._labels.append(self._label_to_index[label_name])
                    self._weights.append(double(1.0))

    def num_classes(self):
        return len(self._label_to_index)

    def label_to_index(self, label):
        if label in self._label_to_index.keys():
            return self._label_to_index[label]
        return -1

    def index_to_label(self, class_index):
        if class_index in self._index_to_label.keys():
            return self._index_to_label[class_index]
        return -1

    def __getitem__(self, index):
        image = tiff.imread(self._image_paths[index])
        # replace no_value with mean
        h, w, c = image.shape
        for i in range(c):
            image[..., i][image[..., i] == NODATA_VALUE] = DATASET_MEAN[i]

        if self.stage == "train":
            if np.random.rand() >= 0.5:
                # flip horizontal
                image = image[::-1, ...].copy()
            if np.random.rand() >= 0.5:
                # flip vertical
                image = image[:, ::-1, :].copy()
        
        return self._transform(image), self._labels[index], self._weights[index]

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