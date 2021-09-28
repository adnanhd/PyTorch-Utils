#!/usr/bin/env python3

import os
import torch
import numpy as np
import os.path as osp
from tqdm import tqdm
from .utils2 import Utils, bezier_curve as bc
from airfoils import Airfoil
import matplotlib.pyplot as plt

bezier_curve = lambda x_points, degree: bc(x_points, degree, Utils.Point.linspace)

class Datum:
    def __init__(self, name, _input, _ground, _output):
        self._name = name
        self.inp = _input
        self.out = _output
        self.gnd = _ground

    @staticmethod
    def to_gpu(sample_image):
        if type(sample_image) != np.ndarray:
            return sample_image.squeeze().gpu().detach().numpy()
        else:
            return sample_image.squeeze()

    @staticmethod
    def to_cpu(sample_image):
        if type(sample_image) != np.ndarray:
            return sample_image.squeeze().cpu().detach().numpy()
        else:
            return sample_image.squeeze()

    @staticmethod
    def to_bezier_curve(points):
        curve_x = bezier_curve(points[0], points.shape[1] - 1)
        curve_y = bezier_curve(points[1], points.shape[1] - 1)
        if isinstance(curve_x, numpy.ndarray):
            curve_y = torch.from_numpy(curve_y)
            curve_x = torch.from_numpy(curve_x)
        return torch.cat((curve_x.unsqueeze(0), curve_y.unsqueeze(0)), dim=0)

    @property
    def mse(self):
        if isinstance(self.out, torch.Tensor):
            return torch.mean(torch.square(self.gnd - self.out))
        else:
            return np.mean(np.square(self.gnd - self.out))

    @property
    def mse_norm(self):
        return np.mean(np.square(self.gnd - self.out) / np.abs(self.gnd)) * 100

    @staticmethod
    def _subplot_img(img, fig=None, ax=None, title=None, color=None, label=None):
        if not ax:
            fig, ax = plt.subplots()

        if title:
            ax.set_title(title)
        
        if not isinstance(img, torch.Tensor):
            image = torch.from_numpy(img)
        else:
            image = img.detach().cpu()
        mask = torch.ones(image.shape)
        masked = torch.logical_and(mask, image)
        
        conf = ax.contourf(masked, cmap=plt.cm.jet)
        fig.colorbar(conf, ax=ax, label=label)

    @staticmethod
    def _subplot_pts(x, y, ax=None, title=None, color=None, label=None):
        if not ax:
            ax = plt.figure().add_subplot()
        if title:
            ax.set_title(title)
        ax.plot(x, y, color=color, label=label)
        # ax.set_xlim(-0.25, 1.25)  # +- 0.1
        # ax.set_ylim(-0.25, 0.25)  # +- 0.2
        ax.legend()

    @staticmethod
    def _subplot_bzr(bzr, ax=None, title=None, color=None, label=None):
        if not ax:
            ax = plt.figure().add_subplot()
        if title:
            ax.set_title(title)
        ax.scatter(bzr[0].tolist(), bzr[1].tolist(), color=color, label=label)
        ax.legend()

    @staticmethod
    def _subplot_bzr_curve(bzr, title=None, ax=None, color=None, label=None):
        if not ax:
            ax = plt.figure().add_subplot()
        if title:
            ax.set_title(title)
       
        curve_x = bezier_curve(bzr[0], bzr.shape[1] - 1)
        curve_y = bezier_curve(bzr[1], bzr.shape[1] - 1)
        
        ax.plot(curve_x, curve_y, color=color, label=label)
        ax.legend()
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(Utils.Image.y_min, Utils.Image.y_max)  # +- 0.2
        # out_bezier.plot(self.inp.shape[1], color=color, ax=ax)

    def plot(self, path=None, title=None):
        raise NotImplementedError

    def __repr__(self):
        return '<Datum (name="{}")>'.format(self._name)


class __Datum:
    def __init__(self, filename):
        self._path, self._name = os.path.split(filename)

    @staticmethod
    def norm(p1, p2):
        return np.sqrt(np.sum(np.square(p1 - p2)))

    @staticmethod
    def distance(l_st, l_en, pt):
        return np.abs(np.cross(l_en - l_st, pt - l_st) / np.sqrt(np.sum(np.square(l_en - l_st))))

    @property
    def _img(self):
        return np.load(os.path.join(self._path, self._name) + '.img.npy')

    @property
    def _pts(self):
        return np.load(os.path.join(self._path, self._name) + '.pts.npy')

    @property
    def _u_bzr(self):
        return np.load(os.path.join(self._path, self._name) + '.u_bzr.npy')

    @property
    def _l_bzr(self):
        return np.load(os.path.join(self._path, self._name) + '.l_bzr.npy')

    @property
    def _u_cf(self):
        return np.load(os.path.join(self._path, self._name) + '.u_cf.npy')

    @property
    def _l_cf(self):
        return np.load(os.path.join(self._path, self._name) + '.l_cf.npy')

    @property
    def path(self):
        return self._path

    @property
    def name(self):
        return self._name

    def plot_pts(self, ax=None):
        
        if not ax:
            ax = plt.figure().add_subplot()
        ax.scatter(Utils.Point.linspace, self._pts[0], color='red')
        ax.scatter(Utils.Point.linspace, self._pts[1], color='orange')

    def plot_img(self, ax=None):
        img = torch.from_numpy(self._img)
        mask = torch.ones(Utils.Image.shape)
        masked = torch.logical_and(mask, img.squeeze())

        if not ax:
            ax = plt.figure().add_subplot()
        ax.contourf(masked, cmap=plt.cm.jet)

    def plot_bzr(self, ax=None):
        u_bzr = self._u_bzr
        l_bzr = self._l_bzr

        if not ax:
            ax = plt.figure().add_subplot()
        ax.scatter(u_bzr[0], u_bzr[1], color='blue')
        ax.scatter(l_bzr[0], l_bzr[1], color='cyan')
    
    def plot_cf(self, ax=None):
        u_cf = self._u_cf
        l_cf = self._l_cf

        if not ax:
            ax = plt.figure().add_subplot()
        ax.plot(Utils.SkinFriction.linspace, u_cf, color='green')
        ax.plot(Utils.SkinFriction.linspace, l_cf, color='yellow')

    def plot(self, title=None, path=None):
        fig = plt.figure(figsize=(22, 11))

        if not title:
            title = self._name.split('.')[0]

        plt.title(title)

        try:
            self.plot_bzr(fig.add_subplot(4, 1, 1))
            self.plot_pts(fig.add_subplot(4, 1, 2))
            self.plot_img(fig.add_subplot(4, 1, 3))
            self.plot_cf (fig.add_subplot(4, 1, 4))
        except FileNotFoundError:
            pass

        if path:
            plt.savefig(path)
        else:
            plt.show()
        plt.close(fig)

    @classmethod
    def load_data_from(cls, root):
        # ../Data/TRAIN
        return [cls(filename=osp.join(root, dirs, dirs))
                for dirs in tqdm(os.listdir(root))
                if osp.isfile(osp.join(root, dirs, dirs + '.img.npy'))
                and osp.isfile(osp.join(root, dirs, dirs + '.u_bzr.npy'))
                and osp.isfile(osp.join(root, dirs, dirs + '.l_bzr.npy'))
                and osp.isfile(osp.join(root, dirs, dirs + '.pts.npy'))
                and osp.isfile(osp.join(root, dirs, dirs + '.u_cf.npy'))
                and osp.isfile(osp.join(root, dirs, dirs + '.l_cf.npy'))]

    def __repr__(self):
        return '<ShallowDatum (name="{}")>'.format(self._name)


def load_data_from(filepath='../NewData2/TRAIN'):
    return __Datum.load_data_from(root=filepath)

