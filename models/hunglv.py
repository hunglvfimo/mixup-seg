import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from torchsummary import summary

cfg = {
    '11': [64, 128, 256, 256, 512, 512, 512, 512],
    '13': [64, 64, 128, 128, 256, 256, 512, 512, 512, 512],
    '16': [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512],
    '19': [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512],
}

class HungNet(nn.Module):
    def __init__(self, name, in_channels, n_classes):
		super(HungNet, self).__init__()
		self.features 	= self._make_layers(cfg[name], in_channels)
		self.classifier = nn.Linear(512, n_classes)

	def forward(self, x):
		out = self.features(x)
		out = F.max_pool2d(out, kernel_size=out.size()[2:])
		out = out.view(out.size(0), -1)
		out = self.classifier(out)
		return out

	def _make_layers(self, cfg, in_channels):
	    layers = []
	    for x in cfg:
	    	layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
	    	in_channels = x
		return nn.Sequential(*layers)

if __name__ == '__main__':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	debug_model = HungNet("11", 3, 9).to(device)
	summary(debug_model, (3, 3, 3))