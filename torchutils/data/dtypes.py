import os
import torch
import hashlib
import scipy.io
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class Datum(ABC): 
    # must be overloaded in the derived class
    shape = None 
    suffix = None
    # remove from class
    sha256 = hashlib.sha256()

    def __init__(self, data=None):
        # overload as assign
        self.data = data

    @property
    def dtype(self):
        if isinstance(self, Datum):
            return type(self)
        elif isinstance(self.data, torch.Tensor) \
                or isinstance(self.data, np.ndarray):
            return self.data.dtype
        else:
            return type(self.data)

    @property
    def datum(self):
        if isinstance(self.data, torch.Tensor):
            return self.data.clone()
        elif isinstance(self.data, np.ndarray):
            return self.data.copy()
        else:
            return deepcopy(self.data)

    # TODO: add setters of dtype and device like to_dtype and to_device
    def to_device(self, device):
        if not isinstance(self.data, torch.Tensor):
            data = torch.as_tensor(self.data)
        else:
            data = self.data

        return data.to_device(device).clone()

    # TODO: change these methods so that these methods outputs an array of 
    # Tensor or NumPy instead of a Datum object
    def numpy(self):
        return Datum(data=self.np.asarray(self.datum))

    def tensor(self):
        return Datum(data=torch.as_tensor(self.datum))
    
    def gpu(self):
        assert isinstance(self.data, torch.Tensor)
        return Datum(data=self.datum.cuda())

    def cpu(self):
        assert isinstance(self.data, torch.Tensor)
        return Datum(data=self.datum.cpu())

    @abstractmethod
    def subplot(self, fig=None, ax=None, title=None, color=None, label=None):
        raise NotImplementedError

    @classmethod
    def load(cls, path, name):
        path = os.path.join(path, name, name + cls.suffix)
        return cls(torch.from_numpy(np.load(path)))

    @classmethod
    def hash(cls):
        digest = f'{cls.suffix}${"@".join(map(str, cls.shape))}'
        return hashlib.sha256(bytes(digest, encoding='utf-8')).hexdigest()

    def __hash__(self):
        return int(self.__class__.hash(), 16)

    def __iter__(self):
        return iter(self.data)


class __TestDatum(Datum):
    shape = (2, 70)
    suffix = '.tst.npy'
    y_part= +.4
    __slice = (shape[1] // 7) * 4

    linspace = np.append(torch.linspace(0.0, y_part, __slice + 1)[
                               :__slice], torch.linspace(y_part, 1, shape[1] - __slice))

    def __init__(self, pts=None):
        super(self.__class__, self).__init__()
        if pts is None:
            pts=np.zeros(self.__class__.shape)
        self.data = pts.reshape(self.__class__.shape)
    
    def subplot(self, fig=None, ax=None, title=None, color=None, label=None):
        if not ax:
            fig, ax = plt.subplots()
        if title:
            ax.set_title(title)
        ax.plot(self.__class__.linspace, self.data, color=color, label=label)
        ax.legend()

        return fig, ax
