#!/usr/bin/env python3
import torch
import torch.nn as nn
from .mlp import FullyConnectedBlock
from .cnn import ConvolutionBlock

class AutoencoderMLP(nn.Module):
    def __init__(self, li=3136, l1=128, l2=128, l3=128, lo=1000, init_weight=None):
        super(AutoencoderMLP, self).__init__()
        self.li = li

        self.conv_block = ConvolutionBlock(1, 4, 16, 64, kernel_size=4, maxpool=(3, 3), activation=nn.ReLU()) 
        self.features = nn.Linear(li, 16)
        self.fcb = FullyConnectedBlock(16, l1, l2, l3, lo, activation=nn.ReLU())

    def forward(self, x):                     ## [None, 1, 216, 216]
        x = self.conv_block(x)
        x = x.view(-1, self.li)
        x = self.features(x)

        return self.fcb(x)

