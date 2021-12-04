#!/usr/bin/env python3.6
from .mlp2 import MultiLayeredPerceptron
from .cnn import ConvolutionNeuralNetwork, ConvolutionBlock
from .ausm_model import CFD_CNN as AUSM_CFD_CNN
from .auto_cnn import AutoencoderCNN
from .auto_mlp import AutoencoderMLP


def BezierMLP(li=70, l1=256, l2=128, l3=64, lo=12):
    return MultiLayeredPerceptron(li, l1, l2, l3, lo)


def SurfaceMLP(li=70, l1=128, l2=512, lo=1000):
    return MultiLayeredPerceptron(li, l1, l2, lo)


__version__ = "1.0.0"

