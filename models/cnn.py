#!/usr/bin/env python3
import torch
import torch.nn as nn
from .mlp2 import MultiLayeredPerceptron


class ConvolutionBlock(nn.Module):
    def __init__(self, *argv, kernel_size, maxpool=(3, 3), activation=nn.ReLU()):
        assert len(argv) > 1
        super(ConvolutionBlock, self).__init__()

        layers = []
        for i in range(len(argv) - 1):
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels=argv[i],
                          out_channels=argv[i+1],
                          kernel_size=kernel_size,
                          stride=1,
                          padding=1),
                nn.BatchNorm2d(argv[i+1]),
                activation,
                nn.MaxPool2d(*maxpool))
            )

        self.seq = nn.Sequential(*layers)


    def forward(self, x):                     ## [None, 1, 216, 216]
        return self.seq(x)


def ConvolutionNeuralNetwork(*argv, **kwargs):
    return nn.Sequential(
            ConvolutionBlock(*argv[:-1], **kwargs),
            nn.Linear(argv[-2], argv[-1])
            )
