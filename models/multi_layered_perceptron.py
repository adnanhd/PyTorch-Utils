import torch
import torch.nn as nn


class MultiLayeredPerceptron2D(nn.Module):
    def __init__(self, in_channel, out_channel, 
                        l1_channel, l2_channel):
        super(MultiLayeredPerceptron2D, self).__init__()

        self.fc_1 = nn.Linear(in_channel, l1_channel)
        self.fc_2 = nn.Linear(l1_channel, l2_channel)
        self.output_layer = nn.Linear(l2_channel, out_channel)
        
        self.activation = nn.ReLU()
        
        self.bn_1 = nn.BatchNorm1d(l1_channel)
        self.bn_2 = nn.BatchNorm1d(l2_channel)
        self.bn_3 = nn.BatchNorm1d(out_channel)
        
    def forward(self, x):   ##256x256,1

        x = self.fc_1(x)
        x = self.bn_1(x)
        x = self.activation(x)

        x = self.fc_2(x)
        x = self.bn_2(x)
        x = self.activation(x)

        return self.output_layer(x)


class MultiLayeredPerceptron3D(nn.Module):
    def __init__(self, in_channel, out_channel, 
            l1_channel, l2_channel, l3_channel):
        super(MultiLayeredPerceptron3D, self).__init__()

        self.fc_1 = nn.Linear(in_channel, l1_channel)
        self.fc_2 = nn.Linear(l1_channel, l2_channel)
        self.fc_3 = nn.Linear(l2_channel, l3_channel)
        self.output_layer = nn.Linear(l3_channel, out_channel)
        
        self.activation = nn.ReLU()
        
        self.bn_1 = nn.BatchNorm1d(l1_channel)
        self.bn_2 = nn.BatchNorm1d(l2_channel)
        self.bn_3 = nn.BatchNorm1d(l3_channel)
        self.bn_4 = nn.BatchNorm1d(out_channel)
        
    def forward(self, x):   ##256x256,1

        x = self.fc_1(x)
        x = self.bn_1(x)
        x = self.activation(x)

        x = self.fc_2(x)
        x = self.bn_2(x)
        x = self.activation(x)

        x = self.fc_3(x)
        x = self.bn_3(x)
        x = self.activation(x)

        return self.output_layer(x)
