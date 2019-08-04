import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from torchsummary import summary

cfg = {
	'11': [32, 'M', 64, 'M'],
}

class Flatten(nn.Module):
	def forward(self, input):
		return input.view(input.size(0), -1)

class HungNet(nn.Module):
	def __init__(self, name, in_channels, n_classes):
		super(HungNet, self).__init__()
		self.features 	= self._make_layers(cfg[name], in_channels)
		self.flatten 	= Flatten()
		self.classifier = nn.Linear(256, n_classes)

	def forward(self, x):
		out = self.features(x)
		out = self.flatten(out)
		out = self.classifier(out)
		return out

	def _make_layers(self, cfg, in_channels):
		layers = []
		for x in cfg:
			if x == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			else:
				layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
				in_channels = x
		return nn.Sequential(*layers)

def HungNet11(in_channels, n_classes):
	return HungNet("11", in_channels, n_classes)

if __name__ == '__main__':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	debug_model = HungNet11(21, 9).to(device)
	summary(debug_model, (21, 9, 9))