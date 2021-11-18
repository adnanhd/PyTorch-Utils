import os
import math
import torch
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from .dtypes import Datum
from .sample import StoredSample, Sample
from .utils import Constants, generate_dataset

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
    def to_dataloader(cls, features, labels, batch_size=1, train=True, transform=None, num_workers=0):
        return torch.utils.data.DataLoader(cls(features, labels, transform=transform),
                                           batch_size=batch_size,
                                           shuffle=train,
                                           num_workers=num_workers)

    def dataloader(self, batch_size=1, train=True, num_workers=0):
        return torch.utils.data.DataLoader(self, shuffle=train,
                                           batch_size=batch_size,
                                           num_workers=num_workers)

    @classmethod
    def from_preloaded(cls, feature_class, label_class, 
            transform=None, feature_func=None, label_func=None, 
            path=Constants.preloaded_dataset_path, cache=True):
        
        try:
            import tqdm
        except ImportError:
            iterator = range
        else:
            iterator = tqdm.tqdm

        cache_filepath = hex(int(label_class.hash(), 16) ^ int(feature_class.hash(), 16))[2:]
        cache_filepath = os.path.join(path, cache_filepath)

        if os.path.isfile(cache_filepath):
            cache = torch.load(cache_filepath)
            features, labels = cache['features'], cache['labels']
        else:
            datapath = list(generate_dataset(feature_class, label_class, filepath=path))
            
            features = torch.empty(torch.Size([len(datapath)]) + torch.Size(feature_class.shape))
            labels = torch.empty(torch.Size([len(datapath)]) + torch.Size(label_class.shape))
            
            for i, datum_name in enumerate(iterator(datapath)):
                features[i] = feature_class.load(path, datum_name).data
                labels[i] = label_class.load(path, datum_name).data
            
            if cache:
                torch.save({'features': features, 'labels': labels}, cache_filepath)

        return cls(features, labels, transform, feature_func, label_func)



