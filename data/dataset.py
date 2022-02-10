import imp
import os
import math
import utils
import utils.data

from importlib_metadata import version
import torch
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from .dtypes import Datum
from .sample import StoredSample, Sample
from .utils import Constants, generate_dataset

__version__ = '1.0.a'


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
        label = self.labels[index]

        if isinstance(feature, Datum):
            feture = feature.data
        if isinstance(label, Datum):
            label = label.data

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
        if isinstance(index, float):
            index = int(self.__len__() * index)
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

    @property
    def shape(self):
        return self.labels.shape

    @classmethod
    def to_dataloader(cls, features, labels, batch_size=None, train=True, transform=None, num_workers=0):
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

    @classmethod
    def load_from(cls, path):
        data = torch.load(path)
        return cls(features=data['features'], labels=data['labels'])

    def load(self, path):
        data = torch.load(path)
        self.labels = data['labels']
        self.features = data['features']

    def save(self, path):
        torch.save({'features': self.features, 'labels': self.labels,
                    'utils.__version__': utils.__version__,
                    'utils.data.__version__': utils.data.__version__,
                    'utils.data.dataset.__version__': __version__, }, path)

    def __repr__(self):
        return f"<Dataset: {self.__len__()} samples>(features: {self.features.shape}, labels: {self.labels.shape})"
