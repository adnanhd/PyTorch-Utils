#!/usr/bin/env python3

import os
import torch
import numpy as np
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt
from .utils import generate_dataset


class Sample:
    def __init__(self, name, _input, _ground, _output=None):
        self._name = name
        self.inp = _input
        self.out = _output
        self.gnd = _ground
    
    """
    @staticmethod
    def to_bezier_curve(points):
        curve_x = bezier_curve(points[0], points.shape[1] - 1)
        curve_y = bezier_curve(points[1], points.shape[1] - 1)
        if isinstance(curve_x, numpy.ndarray):
            curve_y = torch.from_numpy(curve_y)
            curve_x = torch.from_numpy(curve_x)
        return torch.cat((curve_x.unsqueeze(0), curve_y.unsqueeze(0)), dim=0)


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
    """

    def plot(self, fig=None, path=None, title=None):
        if fig:
            ax1, ax2 = fig.subplots(2)
        else:
            fig, (ax1, ax2) = plt.subplots(2, figsize=(11, 11))

        if not title:
            title = osp.split(self._name)[1].upper().replace('_', ' with ')

        fig.suptitle(title)
        self.inp.subplot(ax=ax1, fig=fig, color='blue')
        self.gnd.subplot(ax=ax2, fig=fig, color='red')

        if not isinstance(self.out, type(None)):
            self.out.subplot(ax=ax2, fig=fig, color='orange')

        if path:
            plt.savefig(path)
        else:
            plt.show()

        plt.close(fig)

    def __repr__(self):
        return '<Datum (name="{}")>'.format(self._name)
    
    @classmethod
    def from_datum(cls, input_class, ground_class, filepath=None):
        for datum_folder in generate_dataset(input_class, ground_class, filepath=filepath):
            try:
                yield cls(name=datum_folder,
                        _input=input_class.load(filepath, datum_folder),
                        _ground=ground_class.load(filepath, datum_folder)
                    )
            except FileNotFoundError:
                continue


class StoredSample:
    def __init__(self, filename=None, path=None, name=None):
        if filename:
            self._path, self._name = os.path.split(filename)
        self._path = path
        self._name = fname

    @staticmethod
    def norm(p1, p2):
        return np.sqrt(np.sum(np.square(p1 - p2)))

    @staticmethod
    def distance(l_st, l_en, pt):
        return np.abs(np.cross(l_en - l_st, pt - l_st) / np.sqrt(np.sum(np.square(l_en - l_st))))

    @property
    def _img(self):
        return Image.load(self._path, self._name).data[:].float()

    @property
    def _pts(self):
        return Point.load(self._path, self._name).data[:]

    @property
    def _u_bzr(self):
        return Bezier.load(self._path, self._name).data[0]

    @property
    def _l_bzr(self):
        return Bezier.load(self._path, self._name).data[1]

    @property
    def _u_cf(self):
        return SkinFriction.load(self._path, self._name).data[0]

    @property
    def _l_cf(self):
        return SkinFriction.load(self._path, self._name).data[1]

    @property
    def path(self):
        return self._path

    @property
    def name(self):
        return self._name
    
    """
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
    """
    def __repr__(self):
        return '<ShallowDatum (name="{}")>'.format(self._name)
    
    @classmethod
    def from_datum(cls, *datum_classes, filepath=None):
        for datum_folder in generate_dataset(feature_class, label_class, filepath=filepath):
            yield cls(path=filepath, name=datum_folder)

