#!/usr/bin/env python3

import torch
import torch.nn as nn

class MultiLayeredPerceptron(nn.Module):
    def __init__(self, *argv, init_weight=None, activation=nn.ReLU(), **kwargs):
        assert len(argv) > 1
        super(MultiLayeredPerceptron, self).__init__()

        layers = []
        for i in range(len(argv) - 2):
            fc_layer = nn.Linear(argv[i], argv[i+1])

            if init_weight is not None:
                init_weight(fc_layer.weight)

            layers.append(nn.Sequential(
                fc_layer,
                nn.BatchNorm1d(argv[i+1]),
                activation)
            )

        output_layer = nn.Linear(argv[-2], argv[-1])
        self.seq = nn.Sequential(*layers, output_layer)

    def forward(self, x):
        return self.seq(x)

