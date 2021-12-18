#!/usr/bin/env python3
import torch
import torch.nn as nn


class ConvolutionBlock(nn.Module):
    def __init__(self, *layers, kernel_size, stride=1, padding=0, max_pool=None, activation=nn.ReLU()):
        assert len(layers) > 1
        super(ConvolutionBlock, self).__init__()

        seq_layers = []
        for i in range(len(layers) - 1):
            layer = [
                nn.Conv2d(in_channels=layers[i],
                          out_channels=layers[i+1],
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding),
                nn.BatchNorm2d(layers[i+1]),
                activation]

            if max_pool is not None:
                layer.append(nn.MaxPool2d(*max_pool))
            
            seq_layers.append(nn.Sequential(*layer))

        self.seq = nn.Sequential(*seq_layers)

    def forward(self, x):
        return self.seq(x)


def EncoderBlock(*layers, kernel_size, stride=1, padding=0, activation=nn.ReLU()):
    return ConvolutionBlock(*layers, kernel_size=kernel_size, stride=stride, activation=activation)


class DecoderBlock(nn.Module):
    def __init__(self, *layers, kernel_size, stride=1, padding=0, activation=nn.ReLU()):
        assert len(layers) > 1
        super(DecoderBlock, self).__init__()

        seq_layers = []
        for i in range(len(layers) - 1):
            seq_layers.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels=layers[i],
                                   out_channels=layers[i+1],
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding),
                activation)
                )

        self.seq = nn.Sequential(*seq_layers)

    def forward(self, x):
        return self.seq(x)


def ConvolutionNeuralNetwork(*layers, **kwargs):
    return nn.Sequential(
            ConvolutionBlock(*layers[:-1], **kwargs),
            nn.Linear(layers[-2], layers[-1])
            )
