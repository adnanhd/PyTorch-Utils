import os
import math
import torch
import warnings
import numpy as np
from .dtypes import Datum
import matplotlib.pyplot as plt

class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, features, labels, transform=None, feature_func=None, label_func=None):
        self.features = features
        self.labels = labels
        self.transform = transform
        self.feature_func = feature_func
        self.label_func = label_func

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        feature = self.features[index]
        label   = self.labels[index]

        if isinstance(feature, Datum):
            feture = feature.data
        if isinstance(label, Datum):
            label  = label.data

        if self.feature_func:
            feature = self.feature_func(feature)

        if self.label_func:
            label = self.label_func(label)
        
        if self.transform:
            feature = self.transform(feature)
        
        return feature, label

    def skip(self, index):
        self.features = self.features[index:]
        self.labels = self.labels[index:]

    def take(self, start, end=None):
        if not end:
            end = start
            start = 0
        return Dataset(features=self.features[start:end], 
                labels=self.labels[start:end], 
                transform=self.transform,
                feature_func=self.feature_func,
                label_func=self.label_func)
    
    def leave(self, index):
        warnings.warn('This method is renamed as split in the future versions', FutureWarning)
        return self.split(index)

    def split(self, index, shuffle = False):
        if isinstance(index, float):
            index = int(self.__len__() * index)
        
        try:
            from sklearn.model_selection import train_test_split
            data = train_test_split(self.features, self.labels, test_size=index, shuffle=shuffle)
            self.features = data[0]
            self.labels = data[2]
        
            other = self.take(0)
            other.features = data[1]
            other.labels = data[3]
        except ImportError:
            warnings.warn("sklearn.model_selection could not imported, therefore shuffle feature is not available", ImportWarning)
            other = self.take(index)
            self.skip(index)

        return other

    def map(self, feature_func=None, label_func=None):
        if feature_func:
            self.feature_func = feature_func

        if label_func:
            self.label_func = label_func

        if (not feature_func) and (not label_func):
            self.feature_func = None
            self.label_func = None

    # TODO: add a cache method that stores the dataset as a .npy pickle
    @property
    def shape(self):
        return self.labels.shape

    @property
    def device(self):
        assert self.labels.device == self.features.device, f"features and labels are supposed to be on the same device {self.labels.device} {self.features.device}"
        return self.labels.device

    @classmethod
    def as_dataloader(cls, features, labels, batch_size=None, train=True, transform=None, num_workers=0):
        if batch_size is None:
            batch_size = len(features)
        return torch.utils.data.DataLoader(cls(features, labels, transform=transform),
                                           batch_size=batch_size,
                                           shuffle=train,
                                           num_workers=num_workers)

    def dataloader(self, batch_size=None, train=True, num_workers=0):
        if batch_size is None:
            batch_size = self.__len__()
        return torch.utils.data.DataLoader(self, shuffle=train,
                                           batch_size=batch_size,
                                           num_workers=num_workers)

    def __repr__(self):
        device = self.features.device
        if device != self.labels.device:
            device = (device, self.labels.device)
        return f"<Dataset: {self.__len__()} samples>(features: {self.features.shape}, labels: {self.labels.shape}, device: {device})"

