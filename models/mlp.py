#!/usr/bin/env python3

import torch
import torch.nn as nn

class MultiLayeredPerceptron(nn.Module):
    def __init__(self, *argv, init_weight=None, activation=nn.ReLU(), **kwargs):
        super(MultiLayeredPerceptron, self).__init__()

        layers = []
        if len(argv) >= 2:
            layers.append(nn.Linear(argv[0], argv[1]))


        for i in range(2, len(argv)):
            layers.append(nn.BatchNorm1d(argv[i-1]))
            layers.append(activation)
            layers.append(nn.Linear(argv[i-1], argv[i]))

            if init_weight:
                init_weight(layers[-1].weight)

        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)

