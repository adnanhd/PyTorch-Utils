#!/usr/bin/env python3.6

from .autoencoder_mlp import AutoencoderMLPModel as AutoencoderMLP
from .autoencoder_cnn import AutoencoderCNNModel as AutoencoderCNN
from .ausm_model      import CFD_CNN as AUSM_CFD_CNN

__version__ = "1.0.0"
__all__ = [AutoencoderMLP, AutoencoderCNN, AUSM_CFD_CNN]

