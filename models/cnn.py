#!/usr/bin/env python3
import torch
import torch.nn as nn


class ConvolutionBlock(nn.Module):
    def __init__(self, *argv, kernel_size, stride=1, padding=0, maxpool=None, activation=nn.ReLU()):
        assert len(argv) > 1
        super(ConvolutionBlock, self).__init__()

        layers = []
        for i in range(len(argv) - 1):
            layer = [
                nn.Conv2d(in_channels=argv[i],
                          out_channels=argv[i+1],
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding),
                nn.BatchNorm2d(argv[i+1]),
                activation]

            if maxpool is not None:
                layer.append(nn.MaxPool2d(*maxpool))
            
            layers.append(nn.Sequential(*layer))

        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


def EncoderBlock(*argv, kernel_size, stride=1, padding=0, activation=nn.ReLU()):
    return ConvolutionBlock(*argv, kernel_size, stride=stride, activation=activation)


class DecoderBlock(nn.Module):
    def __init__(self, *argv, kernel_size, stride=1, padding=0, activation=nn.ReLU()):
        assert len(argv) > 1
        super(ConvolutionBlock, self).__init__()

        layers = []
        for i in range(len(argv) - 1):
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels=argv[i],
                                   out_channels=argv[i+1],
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding),
                nn.BatchNorm2d(argv[i+1]),
                activation)
                )

        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


def ConvolutionNeuralNetwork(*argv, **kwargs):
    return nn.Sequential(
            ConvolutionBlock(*argv[:-1], **kwargs),
            nn.Linear(argv[-2], argv[-1])
            )
