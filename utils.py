'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import argparse

import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.transforms as transforms
from dataset import TiffFolder

def online_mean_and_sd(loader, nodata_value=-1):
    """Compute the mean and sd in an online fashion
        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = None
    snd_moment = None

    for (data, _) in loader:
        b, c, h, w = data.shape

        if fst_moment is None:
            fst_moment = torch.empty(c)
        if snd_moment is None:
            snd_moment = torch.empty(c)

        nb_pixels       = b * h * w

        sum_            = torch.empty(c)
        sum_of_square   = torch.empty(c)
        for i in range(c):
            c_data      = data[:, i, ...]
            c_data      = c_data[c_data > nodata_value]

            sum_[i]     = torch.sum(c_data)
            sum_of_square[i] = torch.sum(c_data ** 2)

        fst_moment  = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment  = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt         += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


if __name__ == '__main__':
    parser  = argparse.ArgumentParser()
    parser.add_argument('--dir', default="", type=str, help='')
    args    = parser.parse_args()

    ds          = TiffFolder(args.dir, transform=transforms.Compose([transforms.ToTensor(),]))
    dataloader  = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    mean, std   = online_mean_and_sd(dataloader)
    print(mean)
    print(std)