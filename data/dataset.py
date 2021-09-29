import os
import math
import torch
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
    def from_preloaded(self, feature_class, label_class, 
            transform=None, feature_func=None, label_func=None, 
            path=Constants.preloaded_dataset_path, cache=True):
        def check_nbits(val):
            return math.pow(2, math.floor(math.log2(math.log2(abs(val)))))
        
        def tohex(val, nbits=32):
            nbits = max(nbits, check_nbits(val))
            return hex((val + (1 << nbits)) % (1 << nbits))

        try:
            import tqdm
        except ImportError:
            iterator = range
        else:
            iterator = tqdm.tqdm

        feature_cache = os.path.join(path, ".{}.cache".format(tohex(hash(feature_class))))
        label_cache = os.path.join(path, ".{}.cache".format(tohex(hash(label_class))))
        print(feature_cache, label_cache)
        if os.path.isfile(feature_cache) and os.path.isfile(label_cache):
            features = torch.load(feature_cache)
            labels = torch.load(label_cache)
        else:
            generator = enumerate(iterator(generate_dataset(feature_class, label_class, filepath=path)))

            if os.path.isfile(feature_cache):
                
                features = torch.load(feature_cache)
                labels = torch.empty(torch.Size(label_class.shape))
                for i, datum_name in generator:
                    labels[i] = label_class.load(path, datum_name)
                
                if cache:
                    torch.save(labels, label_cache)
            elif os.path.isfile(label_cache):
                
                features = torch.empty(torch.Size(feature_class.shape))
                labels = torch.load(label_cache)
                for i, datum_name in generator:
                    features[i] = feature_class.load(path, datum_name)
                
                if cache:
                    torch.save(features, feature_cache)
            else:
                
                features = torch.empty(torch.Size([175]) + torch.Size(feature_class.shape))
                labels = torch.empty(torch.Size([175]) + torch.Size(label_class.shape))
                for i, datum_name in generator:
                    features[i] = feature_class.load(path, datum_name).data
                    labels[i] = label_class.load(path, datum_name).data
                
                if cache:
                    torch.save(features, feature_cache)
                    torch.save(labels, label_cache)

        return cls(features, labels, transform, feature_func, label_func)



class PreLoadedDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, feature_class, label_class, 
            path=None, transform=None, 
            datum_names=None, feature_func=None,
            label_func=None):
        super(PreLoadedDataset, self).__init__()
        if not path:
            path = Constants.preloaded_dataset_path

        self.transform = transform
        self.dataset_path = path
        self.feature_class = feature_class
        self.label_class = label_class
        self.feature_func = feature_func
        self.label_func = label_func
        
        if datum_names:
            self.datum_names = datum_names
        else:
            self.datum_names = list(generate_dataset(self.feature_class, 
                self.label_class, filepath=self.dataset_path))

    def __len__(self):
        return len(self.datum_names)

    def __getitem__(self, index):
        feature_datum = self.feature_class.load(self.dataset_path, self.datum_names[index])
        label_datum = self.label_class.load(self.dataset_path, self.datum_names[index])

        feature = feature_datum.data
        label = label_datum.data

        if self.feature_func:
            feature = self.feature_func(feature)

        if self.label_func:
            label = self.label_func(label)

        if self.transform:
            feature = self.transform(feature)

        return feature, label

    def dataset(self, path=Constants.preloaded_dataset_path):
        import tqdm
        features = []
        labels = []
        for feature, label in tqdm.tqdm(self):
            features.append(feature.unsqueeze(dim=0))
            labels.append(label.unsqueeze(dim=0))

        return Dataset(
                features=torch.cat(features), 
                labels=torch.cat(labels), 
                transform=self.transform, 
                feature_func=self.feature_func, 
                label_func=self.label_func)
    
    def dataloader(self, batch_size=1, train=True, num_workers=0):
        return torch.utils.data.DataLoader(self,
                batch_size=batch_size,
                shuffle=train,
                num_workers=num_workers)

    def skip(self, index):
        self.datum_names = self.datum_names[index:]

    def take(self, start, end=None):
        if not end:
            end = start
            start = 0
        return PreLoadedDataset(feature_class=self.feature_class, 
                        label_class=self.label_class, 
                        path=self.dataset_path,
                        transform=self.transform, 
                        datum_names=self.datum_names[start:end],
                        feature_func=self.feature_func,
                        label_func=self.label_func)

    def leave(self, index):
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

