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
            self._index_to_label[index]   = label
            
            for image_path in glob.glob(os.path.join(data_dir, label, "*.tif")):
                self._image_paths.append(image_path)
                self._labels.append(index)

        self._n_classes      = len(list(self._label_to_index))

    def label_to_index(self, label):
        return self._label_to_index[label]

    def index_to_label(self, class_index):
        return self._index_to_label[class_index]

    def __getitem__(self, index):
        image = tiff.imread(self._image_paths[index])
        
        label = np.zeros(self._n_classes)
        label[self._labels[index]] = 1
        
        return self._transform(image), torch.from_numpy(label).long()

    def __len__(self):
        return len(self._image_paths)