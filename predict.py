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

from PIL import Image
import tifffile as tiff
from osgeo import gdal

import models
from dataset import TiffImageSet

from tqdm import tqdm

RANDOM_STATE = 194

DATASET_MEAN    = (1.4113e+03,  6.4466e+03,  6.8021e-02,  3.4658e-02,  2.1277e-02,
			 7.3897e-02,  9.7023e-02,  1.0258e-01,  1.0396e-01,  9.9798e-02,
			 1.0648e-01,  1.0012e-01,  9.5947e-02, -2.4567e-03, -5.7526e-03,
			-7.0414e-02,  5.9038e-02,  2.2282e-02, -8.9304e-02,  1.4858e+00,
			-1.6102e-02)

DATASET_STD     = (1.7253e+03, 8.8444e+03, 1.1058e-02, 3.6809e-02, 2.5553e-02, 1.3737e-02,
			1.8679e-02, 2.8324e-02, 1.7626e-02, 5.1439e-02, 6.2850e-02, 6.8435e-02,
			7.5585e-02, 9.0535e-02, 1.6582e-01, 3.3082e-01, 1.8185e-01, 1.0235e-01,
			3.6490e-01, 7.7676e-01, 1.7835e-01)

parser 		= argparse.ArgumentParser(description='PyTorch Mixup')
parser.add_argument('--snapshot', type=str, default=None)
parser.add_argument('--image_path', type=str, default=None)
parser.add_argument('--save_dir', type=str, default=None)
args        = parser.parse_args()

def predict():
	use_cuda    = torch.cuda.is_available()

	transform   = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(DATASET_MEAN, DATASET_STD),
	])

	checkpoint 	= torch.load(args.snapshot)
	net 		= checkpoint['net']
	rng_state 	= checkpoint['rng_state']
	torch.set_rng_state(rng_state)

	ds 			= TiffImageSet(args.image_path, transform=transform)
	loader 		= torch.utils.data.DataLoader(ds,
											batch_size=128,
											shuffle=False, num_workers=0)

	results 	= np.zeros(ds.get_shape(), dtype=np.uint8)

	if use_cuda:
		net.cuda()
		print('Using CUDA..')

	pbar = tqdm(loader)
	net.eval()
	with torch.no_grad():
		for (ys, xs, inputs) in pbar:
			if use_cuda:
				inputs 	= inputs.cuda()

			inputs 		= Variable(inputs)
			outputs 	= net(inputs)
			y_pred 		= torch.argmax(outputs.data, 1).cpu().numpy()
			for y, x, pred in zip(ys.numpy(), xs.numpy(), y_pred):
				if y >= 0 and x >= 0:
					results[y, x] 	= pred + 1

	src_img 	= gdal.Open(args.image_path)
	trans 		= src_img.GetGeoTransform()
	proj 		= src_img.GetProjection()

	outdriver	= gdal.GetDriverByName("GTiff")
	outdata   	= outdriver.Create(os.path.join(args.save_dir, "pred.tif"), results.shape[1], results.shape[0], 1, gdal.GDT_Byte)
	outdata.GetRasterBand(1).WriteArray(results)
	outdata.GetRasterBand(1).SetNoDataValue(0)
	outdata.SetGeoTransform(trans)
	outdata.SetProjection(proj)

if __name__ == '__main__':
	predict()