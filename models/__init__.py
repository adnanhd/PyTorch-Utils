#!/usr/bin/env python3.6

from .autoencoder_mlp import AutoencoderMLPModel as AutoencoderMLP
from .autoencoder_cnn import AutoencoderCNNModel as AutoencoderCNN
from .ausm_model      import CFD_CNN as AUSM_CFD_CNN
from .multi_layered_perceptron import MultiLayeredPerceptron3D, MultiLayeredPerceptron2D


def BezierMLP(li=70, l1=256, l2=128, l3=64, lo=12):
    return MultiLayeredPerceptron3D(in_channel=li, out_channel=lo, 
                      l1_channel=l1, l2_channel=l2, l3_channel=l3)

def SurfaceMLP(i=70, l1=128, l2=512, lo=1000):
    return MultiLayeredPerceptron2D(in_channel=li, out_channel=lo, 
                                     l1_channel=l1, l2_channel=l2)

def MultiLayeredPerceptron(dim=3):
    assert isinstance(dim, int)
    if dim == 2:
        return MultiLayeredPerceptron2D
    elif dim == 3:
        return MultiLayeredPerceptron333D

__version__ = "2.1.0"
__all__ = [AutoencoderMLP, AutoencoderCNN, AUSM_CFD_CNN, BezierMLP, MultiLayeredPerceptron]

