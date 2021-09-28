#!/usr/bin/envy python3.6

import os

import numpy
import torch
import scipy.io
import numpy as np
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
#from bezier import Curve

from DataPrepare import Utils
from DataPrepare import ShallowDatum
from DataPrepare import Datum

inp_x = Utils().x_sigmoid
y_upper_lim = Utils().y_max
y_lower_lim = Utils().y_min

class MLPDatum(Datum):
    def __init__(self, title, inp, gnd, out, to_numpy=True):
        self.device = inp.device
        if to_numpy:
            inp_y = self.to_cpu(inp)
            inp = np.append(inp_x, inp_y)
            inp.resize(2, 70)

            gnd = self.to_cpu(gnd)
            gnd.resize(2, 6)

            if not isinstance(None, type(out)):
                out = self.to_cpu(out)
                out.resize(2, 6)

        else:
            inp = torch.cat((torch.from_numpy(inp_x).unsqueeze(0), inp.unsqueeze(0)), dim=0)

            if not isinstance(None, type(out)):
                out = out.reshape(2, 6)
            gnd = gnd.reshape(2, 6)

        super(MLPDatum, self).__init__(title, _input=inp, _ground=gnd, _output=out)

    def subplot_input(self, ax=None, color='green'):
        super()._subplot_pts(
                self.inp[0].tolist(), 
                self.inp[1].tolist(), 
                ax=ax, color=color, 
                title='Model Input Points', 
                label='output bezier points')

    def subplot_ground_bezier(self, ax=None, color='green'):
        super()._subplot_bzr_curve(
                self.gnd, ax=ax, color=color, 
                title='Model Input Points', 
                label='input bezier curve')

    def subplot_ground_points(self, ax=None, color='red'):
        super()._subplot_bzr(
                self.gnd, ax=ax, color=color, 
                label='skin friction', title='')

    def subplot_ground_bezier(self, ax=None, color='red'):
        super()._subplot_bzr_curve(
                self.gnd, ax=ax, color=color, 
                label='skin friction', title='')

    def plot(self, path=None, title=None):
        fig, (ax1, ax2) = plt.subplots(2, figsize=(11, 11), sharex='all')

        if not title:
            title = osp.split(self._name)[1].upper().replace('_', ' with ')
        fig.suptitle(title)
        self.subplot_ground_points(ax=ax1, color='red')
        self.subplot_ground_bezier(ax=ax2)
        self.subplot_input(ax=ax2, color='brown')

        if not isinstance(None, type(self.out)):
            self.subplot_output_points(ax=ax2, color='orange')
            self.subplot_output_bezier(ax=ax2)

        if path:
            plt.savefig(path)
        else:
            plt.show()
        plt.close(fig)

    def __repr__(self):
        return '<Datum.MLP (name="{}", device="{}")>'.format(self._name, self.device)


class MLPShallowDatum(ShallowDatum):
    def __init__(self, filename, output=None, is_upper=True):
        super(MLPLoaderNode, self).__init__(filename)
        self.output = output
        self.__is_upper = is_upper

    @property
    def input(self):
        pts = super()._pts
        mid = pts.shape[1] // 2
        if self.__is_upper:
            return torch.from_numpy(super()._pts).squeeze(dim=0)[:mid]
        else:
            return torch.from_numpy(super()._pts).squeeze(dim=0)[mid:]

    @property
    def ground(self):
        if self.__is_upper:
            bzr_x, bzr_y = super()._u_bzr
        else:
            bzr_x, bzr_y = super()._l_bzr
        return torch.from_numpy(np.append(bzr_x, bzr_y)).squeeze(dim=0)

    @property
    def container(self):
        return MLPDatum(self._name, self.input, self.ground, self.output, to_numpy=True)

    def plot(self, title=None, path=None):
        datum = self.container
        datum.plot(title)
        if path:
            plt.savefig(path)
        else:
            plt.show()
        plt.close()

    def sanity_check(self, path=None):
        datum = self.container
        fig, ax = plt.subplots()
        fig.suptitle(self.name)
        datum.subplot_ground_bezier(ax=ax)
        datum.subplot_input(ax=ax)
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
                if osp.isfile(osp.join(root, dirs, dirs + '.u_bzr.npy'))
                and osp.isfile(osp.join(root, dirs, dirs + '.l_bzr.npy'))
                and osp.isfile(osp.join(root, dirs, dirs + '.pts.npy'))]

    def __repr__(self):
        return '<ShallowDatum.MLP (name="{}")>'.format(self._name)


def load_data_from(root):
    return MLPShallowDatum.load_data_from(root=root)

