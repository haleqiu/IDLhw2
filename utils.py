import glob
import os
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class BasicBlock(nn.Module):
    #base_width = 64,
    def __init__(self, in_channels, out_channels, stride = 1, kernel_size = 3, downsample = None, norm_layer = None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        ## TODO padding or not?
        self.conv1 = nn.Conv2d(in_channels, out_channels, stride = stride, 
                               kernel_size=kernel_size, bias=False, padding=1)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        ## the stride should always be 1 to have the same size
        self.conv2 = nn.Conv2d(out_channels, out_channels, stride = 1, 
                               kernel_size=kernel_size, bias=False, padding=1)
        self.bn2 = norm_layer(out_channels)
        
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out