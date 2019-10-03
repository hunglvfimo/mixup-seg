import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from torchsummary import summary

class Flatten(nn.Module):
	def forward(self, input):
		return input.view(input.size(0), -1)

class BasicConv2d(nn.Module):
	def __init__(self, in_channels, out_channels, **kwargs):
		super(BasicConv2d, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
		self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		return F.relu(x, inplace=True)

class InceptionA(nn.Module):
	def __init__(self, in_channels, pool_features):
		super(InceptionA, self).__init__()

		self.branch1x1 		= BasicConv2d(in_channels, 64, kernel_size=1)

		self.branch5x5_1 	= BasicConv2d(in_channels, 48, kernel_size=1)
		self.branch5x5_2 	= BasicConv2d(48, 64, kernel_size=5, padding=2)

		self.branch3x3_1 	= BasicConv2d(in_channels, 64, kernel_size=1)
		self.branch3x3_2 	= BasicConv2d(64, 96, kernel_size=3, padding=1)
		self.branch3x3_3 	= BasicConv2d(96, 96, kernel_size=3, padding=1)

		self.branch_pool 	= BasicConv2d(in_channels, pool_features, kernel_size=1)
	
	def forward(self, x):
		branch1x1 = self.branch1x1(x)

		branch5x5 = self.branch5x5_1(x)
		branch5x5 = self.branch5x5_2(branch5x5)

		branch3x3dbl = self.branch3x3_1(x)
		branch3x3dbl = self.branch3x3_2(branch3x3dbl)
		branch3x3dbl = self.branch3x3_3(branch3x3dbl)

		branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
		branch_pool = self.branch_pool(branch_pool)

		outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
		return torch.cat(outputs, 1)

class ZhangNet(nn.Module):
	def __init__(self, in_channels, n_classes):
		super(ZhangNet, self).__init__()

		self.features 	= self._make_layers(in_channels)
		self.dropout 	= nn.Dropout(0.4)
		self.flatten 	= Flatten()
		self.classifier = nn.Linear(480, n_classes)

	def _make_layers(self, in_channels):
		layers = []

		layers 	+= [BasicConv2d(in_channels, 64, kernel_size=3, padding=1)]
		layers 	+= [nn.MaxPool2d(kernel_size=3, stride=2)]
		layers 	+= [nn.LocalResponseNorm(size=64)]
		layers 	+= [BasicConv2d(64, 128, kernel_size=1)]
		layers 	+= [BasicConv2d(128, 256, kernel_size=3, padding=1)]
		layers 	+= [nn.LocalResponseNorm(size=256)]
		layers 	+= [nn.MaxPool2d(kernel_size=3, stride=2)]
		layers 	+= [InceptionA(256, pool_features=32)] # 224 + 32
		layers 	+= [InceptionA(256, pool_features=64)] # 224 + 64
		layers 	+= [InceptionA(288, pool_features=128)] # 224 + 128
		layers 	+= [InceptionA(352, pool_features=256)] # 224 + 256
		layers 	+= [nn.MaxPool2d(kernel_size=3)]

		return nn.Sequential(*layers)

	def forward(self, x):
		out = self.features(x)
		out = self.dropout(out)
		out = self.flatten(out)
		out = self.classifier(out)

		return out

def ZhangNet15(in_channels, n_classes):
	return ZhangNet(in_channels, n_classes)

if __name__ == '__main__':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	debug_model = ZhangNet(21, 9).to(device)
	summary(debug_model, (21, 15, 15))