import torch
import torch.nn as nn
from .mlp2 import MultiLayeredPerceptron
from .cnn import ConvolutionBlock

class AutoencoderCNN(nn.Module):
    def __init__(self):
        super(AutoencoderCNN, self).__init__()

        ### ENCODER
        self.cnn1 = ConvolutionBlock(1, 32, 32, kernel_size=4, maxpool=(3,3))
        self.cnn2 = ConvolutionBlock(32, 64, 128, kernel_size=4, maxpool=(2,2))
        self.mlp1 = MultiLayeredPerceptron(128, 100, 100, 16)
        self.mlp2 = MultiLayeredPerceptron(16, 100, 100, 140)

    def forward(self, x):  ##216x216,1
        x = self.cnn1(x)
        x = self.cnn2(x)

        x = x.view(-1, 128)
        
        x = self.mlp1(x)
        x = self.mlp2(x)
        
        return x


