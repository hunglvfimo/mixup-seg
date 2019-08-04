#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
from __future__ import print_function

import argparse
import csv
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import models
from dataset import TiffFolder
from tqdm import tqdm

DATASET_MEAN    = (1.4113e+03,  6.4466e+03,  6.8021e-02,  3.4658e-02,  2.1277e-02,
         7.3897e-02,  9.7023e-02,  1.0258e-01,  1.0396e-01,  9.9798e-02,
         1.0648e-01,  1.0012e-01,  9.5947e-02, -2.4567e-03, -5.7526e-03,
        -7.0414e-02,  5.9038e-02,  2.2282e-02, -8.9304e-02,  1.4858e+00,
        -1.6102e-02)

DATASET_STD     = (1.7253e+03, 8.8444e+03, 1.1058e-02, 3.6809e-02, 2.5553e-02, 1.3737e-02,
        1.8679e-02, 2.8324e-02, 1.7626e-02, 5.1439e-02, 6.2850e-02, 6.8435e-02,
        7.5585e-02, 9.0535e-02, 1.6582e-01, 3.3082e-01, 1.8185e-01, 1.0235e-01,
        3.6490e-01, 7.7676e-01, 1.7835e-01)

parser = argparse.ArgumentParser(description='PyTorch Mixup')
parser.add_argument('--train_dir', default="", type=str, help='')
parser.add_argument('--test_dir', default="", type=str, help='')
parser.add_argument('--mixup', help='Use mixup (Default: False)', action='store_true')
parser.add_argument('--lr', default=1e-1, type=float, help='learning rate')
parser.add_argument('--snapshot', type=str, default=None)
parser.add_argument('--model', default="HungNet11", type=str,
                    help='model type (default: HungNet11)')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int,
                    help='total epochs to run')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')
parser.add_argument('--num_workers', default=0, type=int,
                    help='')

args        = parser.parse_args()

use_cuda    = torch.cuda.is_available()

best_acc    = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.seed != 0:
    torch.manual_seed(args.seed)

# Data
print('==> Preparing data..')

transform       = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(DATASET_MEAN,
                         DATASET_STD),
])

trainset    = TiffFolder(args.train_dir, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch_size,
                                          shuffle=True, num_workers=args.num_workers)

testset     = TiffFolder(args.test_dir, transform=transform)
testloader  = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.num_workers)

# Model
if args.snapshot is not None:
    # Load checkpoint.
    pass
else:
    print('==> Building model..')
    net = models.__dict__[args.model](21, 9)

if not os.path.isdir('results'):
    os.mkdir('results')
logname = ('results/log_' + net.__class__.__name__ + '_' + args.name + '_'
           + str(args.seed) + '.csv')

if use_cuda:
    net.cuda()
    print('Using CUDA..')

if args.mixup:
    print('Using mixup')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss  = 0
    correct     = 0
    total       = 0

    pbar        = tqdm(trainloader)
    for batch_idx, (inputs, targets) in enumerate(pbar):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        if args.mixup:
            inputs, targets_a, targets_b, lam   = mixup_data(inputs, targets, args.alpha, use_cuda)
            inputs, targets_a, targets_b        = map(Variable, (inputs, targets_a, targets_b))
        outputs     = net(inputs)
        
        if args.mixup:
            loss        = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            loss        = criterion(outputs, targets)
        train_loss  += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total       += targets.size(0).cpu()

        if args.mixup:
            correct     += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                                + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
        else:
            correct     += predicted.eq(targets.data).cpu().sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description('Loss: %.3f' % loss.item())
    return train_loss / batch_idx, 100. * correct / total

def test(epoch):
    global best_acc
    test_loss   = 0
    correct     = 0
    total       = 0

    pbar        = tqdm(testloader)
    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(pbar):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            
            inputs, targets = Variable(inputs), Variable(targets)
            outputs         = net(inputs)
            loss            = criterion(outputs, targets)

            test_loss       += loss.item()
            
            _, predicted    = torch.max(outputs.data, 1)
            total           += targets.size(0).cpu()
            correct         += predicted.eq(targets.data).cpu().sum()
    
    acc = 100.* correct / total
    if epoch == start_epoch + args.epoch - 1 or acc > best_acc:
        checkpoint(acc, epoch)
    
    if acc > best_acc:
        best_acc = acc
    
    return (test_loss / batch_idx, 100.* correct / total)


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.t7' + args.name + '_'
               + str(args.seed))

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'train acc', 'test loss', 'test acc'])

for epoch in range(start_epoch, args.epoch):
    train_loss, train_acc   = train(epoch)
    test_loss, test_acc     = test(epoch)
    adjust_learning_rate(optimizer, epoch)
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, train_acc, test_loss, test_acc])