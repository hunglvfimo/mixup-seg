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

DATASET_MEAN    = (1.4412e+03,  6.6771e+03,  6.8463e-02,  3.5090e-02,  2.1542e-02,
         7.4142e-02,  9.7441e-02,  1.0294e-01,  1.0430e-01,  1.0020e-01,
         1.0696e-01,  1.0025e-01,  9.6298e-02, -2.6017e-03, -6.0922e-03,
        -7.2946e-02,  6.0688e-02,  2.2337e-02, -9.1976e-02,  1.5068e+00,
        -1.6695e-02)

DATASET_STD     = (1.7444e+03, 8.5088e+03, 9.5835e-03, 3.7184e-02, 2.5516e-02, 1.1907e-02,
        1.6603e-02, 2.6543e-02, 1.4727e-02, 5.1238e-02, 6.2930e-02, 6.8506e-02,
        7.5792e-02, 9.1515e-02, 1.6759e-01, 3.3372e-01, 1.8578e-01, 1.0341e-01,
        3.6783e-01, 7.7687e-01, 1.8006e-01)

parser = argparse.ArgumentParser(description='PyTorch Mixup')
parser.add_argument('--train_dir', default="", type=str, help='')
parser.add_argument('--test_dir', default="", type=str, help='')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--snapshot', type=str, default=None)
parser.add_argument('--model', default="HungNet11", type=str,
                    help='model type (default: HungNet11)')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int,
                    help='total epochs to run')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')
parser.add_argument('--num_workers', default=0, type=int,
                    help='')

args = parser.parse_args()

use_cuda = torch.cuda.is_available()

best_acc = 0  # best test accuracy
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
    reg_loss    = 0
    correct     = 0
    total       = 0

    pbar        = tqdm(trainloader)
    for batch_idx, (inputs, targets) in enumerate(pbar):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets_a, targets_b, lam   = mixup_data(inputs, targets, args.alpha, use_cuda)
        inputs, targets_a, targets_b        = map(Variable, (inputs, targets_a, targets_b))
        outputs     = net(inputs)
        loss        = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        train_loss  += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description('Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
                            % (train_loss / (batch_idx + 1), reg_loss / (batch_idx + 1),
                                100.* correct/total, correct, total))
    return (train_loss / batch_idx, reg_loss / batch_idx, 100.* correct/total)


def test(epoch):
    global best_acc
    net.eval()
    test_loss   = 0
    correct     = 0
    total       = 0

    pbar        = tqdm(testloader)
    for batch_idx, (inputs, targets) in enumerate(pbar):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        pbar.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss/(batch_idx+1), 100.*correct/total,
                        correct, total))
    
    acc = 100.*correct/total
    if epoch == start_epoch + args.epoch - 1 or acc > best_acc:
        checkpoint(acc, epoch)
    if acc > best_acc:
        best_acc = acc
    return (test_loss/batch_idx, 100.*correct/total)


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
        logwriter.writerow(['epoch', 'train loss', 'reg loss', 'train acc', 'test loss', 'test acc'])

for epoch in range(start_epoch, args.epoch):
    train_loss, reg_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    adjust_learning_rate(optimizer, epoch)
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, reg_loss, train_acc, test_loss, test_acc])