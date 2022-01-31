import os
import torch
import hashlib
import scipy.io
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from .utils import encoder_lambda
from abc import ABC, abstractmethod


class Datum(ABC):
    sha256 = hashlib.sha256()
    def __init__(self, data=None):
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

    def to_device(self, device):
        if not isinstance(self.data, torch.Tensor):
            data = torch.as_tensor(self.data)
        else:
            data = self.data

        return data.to_device(device).clone()

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
        return hashlib.sha256(bytes(cls.suffix, encoding='utf-8')).hexdigest()

    def __iter__(self):
        return iter(self.data)


class Image(Datum):
    y_min = -.256
    y_max = +.256 

    shape = (1, 216, 216)
    suffix = '.img.npy'
    head_margin = 26
    tail_margin  = 10

    # spacing into how many useful pixels are there in an image
    # TODO: there may be a bug here
    # usespace = np.linspace(0, 1, shape[1] - head_margin, tail_margin + 1)
    usespace = np.linspace(0, 1, shape[1] - head_margin)
    
    encoder = encoder_lambda(shape[1] - 1, y_min, y_max)

    linspace = torch.linspace(y_min, y_max, shape[1] + 1)

    def __init__(self, img):
        self.data = img.reshape(Image.shape)

    def subplot(self, fig=None, ax=None, title=None, color=None, label=None):
        if not ax:
            fig, ax = plt.subplots()

        if title:
            ax.set_title(title)
        
        if not isinstance(img, torch.Tensor):
            image = torch.from_numpy(self.data)
        else:
            image = self.data.detach().cpu()
        mask = torch.ones(image.shape)
        masked = torch.logical_and(mask, image)
        
        conf = ax.contourf(masked, cmap=plt.cm.jet)
        fig.colorbar(conf, ax=ax, label=label)

        return fig, ax


class Point(Datum):
    shape = (2, 70)
    suffix = '.pts.npy'
    y_part= +.4
    __slice = (shape[1] // 7) * 4

    linspace = np.append(torch.linspace(0.0, y_part, __slice + 1)[
                               :__slice], torch.linspace(y_part, 1, shape[1] - __slice))

    def __init__(self, pts):
        self.data = pts.reshape(Point.shape)
    
    def subplot(self, fig=None, ax=None, title=None, color=None, label=None):
        if not ax:
            fig, ax = plt.subplots()
        if title:
            ax.set_title(title)
        ax.plot(Point.linspace, self.data, color=color, label=label)
        # ax.set_xlim(-0.25, 1.25)  # +- 0.1
        # ax.set_ylim(-0.25, 0.25)  # +- 0.2
        ax.legend()

        return fig, ax


class Bezier(Datum):
    shape = (1, 2, 6)
    linspace = None
    suffix = '.u_bzr.npy'

    def __init__(self, bzr):
        self.data = bzr.reshape(Bezier.shape)

    def subplot(self, fig=None, ax=None, title=None, color=None, label=None):
        if not ax:
            fig, ax = plt.subplots()
        if title:
            ax.set_title(title)
        ax.scatter(self.data[0].tolist(), self.data[1].tolist(), color=color, label=label)
        ax.legend()

        return fig, ax


class FlowField(Datum):
    shape = (6, 512, 512)           
    linspace = None
    suffix = '_flowfield.mat'
    keys = ['Rho', 'U', 'V', 'P', 'T', 'Ma'] # ['P', 'Rho', 'T', 'U', 'V', 'Ma', 'Cp'] # 05.01.2022# 05.01.2022# 05.01.2022# 05.01.2022# 05.01.2022

    def __init__(self, cdf):
        self.data = cdf.reshape(FlowField.shape)

    @classmethod
    def load(cls, path, name):
        path = os.path.join(path, name, name + cls.suffix)
        
        y = scipy.io.loadmat(path)
        return cls(torch.stack([torch.from_numpy(y[key]) for key in cls.keys], dim=0))
    
    def subplot(self, fig=None, axes=None, title=None, color=None, label=None):
        if fig is None:
            fig = plt.figure()
        
        if axes is None:
            axes = fig.subplots(self.shape[0])
        else:
            assert all(map(lambda ax: isinstance(ax, plt.Axes), axes))
            
        if title:
            ax1.set_title(title)

        if isinstance(axes, plt.Axes):
            axes = tuple([axes])

        for data, ax, key in zip(self.data, axes, self.keys):
            ax.set_title(key)
            
            if isinstance(self.data, torch.Tensor):
                data = data.detach().cpu().numpy()
                
            fig.colorbar(ax.contourf(data, cmap=plt.cm.jet), ax=ax)

        return fig, axes

class DistFunc(Datum):
    shape = (1, 512, 512)
    linspace = None
    suffix = '_distFunc.mat'

    def __init__(self, cdf):
        assert cdf.shape == self.__class__.shape
        self.data = cdf.reshape(DistFunc.shape)
    
    @classmethod
    def load(cls, path, name=None):
        if name is not None:
            path = os.path.join(path, name, name + cls.suffix)
        
        x = scipy.io.loadmat(path)
        DF = torch.from_numpy(x['DF'])
        return cls(DF.unsqueeze(dim=0))
    
    def subplot(self, fig=None, ax=None, title=None, color=None, label=None):
        if not fig:
            fig = plt.figure()
        
        if not ax:
            ax = fig.subplots()

        if title:
            ax1.set_title(title)

        if isinstance(self.data, torch.Tensor):
            data = self.data[0].detach().cpu().numpy()
        else:
            data = self.data[0]
            
        fig.colorbar(ax.contourf(data, cmap=plt.cm.jet), ax=ax)

        return fig, ax

class SkinFriction(Datum):
    shape = (1, 1000)
    suffix = '.u_cf.npy'
    linspace = torch.linspace(0, 1, shape[1])

    def __init__(self, sfc):
        super(SkinFriction, self).__init__(sfc.reshape(SkinFriction.shape))

    def subplot(self, fig=None, ax=None, title=None, color=None, label=None):
        return fig, ax

