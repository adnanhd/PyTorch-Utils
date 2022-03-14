import imp
import os
import math
import torchutils
import torchutils.data

import torch
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from .dtypes import Datum
from .utils import hybridmethod
from sklearn.model_selection import train_test_split

__version__ = '2.0.a'


class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, features, labels, transform=None, feature_func=None, label_func=None):
        self.features = features
        self.labels = labels
        self.transform = transform
        self.feature_func = feature_func
        self.label_func = label_func

    def __len__(self):
        if len(self.features) != len(self.labels):
            raise IndexError(f'features and labels expected to be same size but found {self.features.shape}, {self.labels.shape}')
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

    def drop(self, start, end=None):
        if end is None:
            end = start
            start = 0
        del self.features[start:end]
        del self.labels[start:end]

    def take(self, start, end=None):
        if end is None:
            end = start
            start = 0
        return Dataset(features=self.features[start:end],
                       labels=self.labels[start:end],
                       transform=self.transform,
                       feature_func=self.feature_func,
                       label_func=self.label_func)

    def train_test_drop(self, index):
        if isinstance(index, float):
            index = int(self.__len__() * index)
        other = self.take(index)
        self.skip(index)
        return other
    
    def split(self, test_size, *args, **kwargs):
        self.features, X_test, self.labels, y_test \
            = train_test_split(self.features, self.labels, 
                    test_size=test_size, *args, **kwargs)

        return Dataset(features=X_test, labels=y_test,
                       transform=self.transform,
                       feature_func=self.feature_func,
                       label_func=self.label_func)

    def train_test_split(self, test_size, valid_size=None, *args, **kwargs):
        test_dataset = self.split(test_size, *args, **kwargs)
        if valid_size is None:
            return test_dataset
        valid_dataset = self.split(valid_size, *args, **kwargs)
        return test_dataset, valid_dataset

    def map(self, feature_func=None, label_func=None):
        flag = True

        if callable(feature_func):
            self.feature_func = feature_func
            flag = False

        if callable(label_func):
            self.label_func = label_func
            flag = False

        if flag:
            self.feature_func = None
            self.label_func = None

    @hybridmethod
    def dataloader(cls, features, labels, batch_size=None, train=True, transform=None, num_workers=0):
        if batch_size is None:
            batch_size = len(features)
        return torch.utils.data.DataLoader(cls(features, labels, transform=transform),
                                           batch_size=batch_size,
                                           shuffle=train,
                                           num_workers=num_workers)

    @dataloader.instancemethod
    def dataloader(self, batch_size=None, train=True, num_workers=0):
        if batch_size is None:
            batch_size = self.__len__()
        return torch.utils.data.DataLoader(self, shuffle=train,
                                           batch_size=batch_size,
                                           num_workers=num_workers)

    @staticmethod
    def _load(path, **kwargs):
        data = torch.load(path, **kwargs)
        def _check_checksum(data, checksum):
            assert checksum['sha256'] == hashlib.sha256(bytes(data)).hexdigest(), "SHA256 checksum must match"
            assert checksum['blake2b'] == hashlib.blake2b(bytes(data)).hexdigest(), "blake2d checksum must match"
            return data

        return _check_checksum(**data['features']), _check_checksum(**data['labels'])

    @hybridmethod
    def load(cls, path, **kwargs):
        features, labels = cls._load(path, **kwargs)
        return cls(features=features, labels=labels)

    @load.instancemethod
    def load(self, path, **kwargs):
        self.features, self.labels = self._load(path, **kwargs)

    def save(self, path):
        def _add_checksum(arr):
            return {'data': arr, 'checksum': {'sha256': hashlib.sha256(bytes(arr)).hexdigest(), 'blake2b': hashlib.blake2b(bytes(arr)).hexdigest()}}

        torch.save({
                'features': _add_checksum(self.features), 
                'labels': _add_checksum(self.labels),
                'package vesion': torchutils.__version__,
                'data module version': torchutils.data.__version__,
                'dataset class version': torchutils.data.dataset.__version__, 
            }, path)

    def __repr__(self):
        return f"<Dataset: {self.__len__()} samples>(features: {self.features.shape}, labels: {self.labels.shape})"
