#!/usr/bin/envy python3.6

import os
import sys

import numpy
import torch
import scipy.io
import numpy as np
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset

from DataPrepare import Utils
from DataPrepare import ShallowDatum
from DataPrepare import Datum
from DataPrepare import MatLoader


class AutoencoderMLPDatum(Datum):
    def __init__(self, title, inp, gnd, out, to_numpy=True):
        self.device = inp.device
        if to_numpy:
            inp = self.to_cpu(inp)
            inp.resize(Utils.Image.shape)

            gnd = self.to_cpu(gnd)
            gnd.resize(Utils.SkinFriction.shape)

            if not isinstance(None, type(out)):
                out = self.to_cpu(out)
                out.resize(Utils.SkinFriction.shape)
            
        else:
            inp = torch.from_numpy(inp)
            inp = inp.reshape(Utils.Image.shape)
            
            gnd = gnd.reshape(Utils.SkinFriction.shape)
            
            if not isinstance(None, type(out)):
                out = out.reshape(Utils.SkinFriction.shape)

        super(AutoencoderMLPDatum, self).__init__(title, _input=inp, _ground=gnd, _output=out)

    def subplot_input(self, fig=None, ax=None, color='blue'):
        super()._subplot_img(self.inp,  
                fig=fig, ax=ax, color=color,
                title='Model Input Image', 
                label='input points')

    def subplot_output(self, ax=None, color='red'):
        super()._subplot_pts(
                Utils.SkinFriction.linspace, 
                self.out.squeeze(), 
                ax=ax, color=color, 
                label='output skin friction')

    def subplot_ground(self, ax=None, color='orange'):
        super()._subplot_pts(
                Utils.SkinFriction.linspace, 
                self.gnd.squeeze(), 
                ax=ax, color=color, 
                label='ground skin friction',
                title='')
    
    def plot(self, path=None, title=None):
        fig, (ax1, ax2) = plt.subplots(2, figsize=(11, 11))

        if not title:
            title = osp.split(self._name)[1].upper().replace('_', ' with ')
        fig.suptitle(title)
        self.subplot_input(ax=ax1, fig=fig, color='blue')
        self.subplot_ground(ax=ax2, color='red')
        #ax2.set_ylim(0, 0.01)

        if not isinstance(None, type(self.out)):
            self.subplot_output(ax=ax2, color='orange')

        if path:
            plt.savefig(path)
        else:
            plt.show()
        plt.close(fig)

    def __repr__(self):
        return '<Datum.AutoencoderMLP (name="{}", device="{}")>'.format(self._name, self.device)


class AutoencoderMLPShallowDatum(ShallowDatum):
    def __init__(self, filename, output=None, is_upper=True):
        super(AutoencoderMLPShallowDatum, self).__init__(filename)
        self.output = output
        self.__is_upper = is_upper

    @property
    def input(self):
        return torch.from_numpy(super()._img).double().unsqueeze(dim=0)

    @property
    def ground(self):
        return torch.from_numpy(super()._u_cf if self.__is_upper else super()._l_cf)

    @property
    def container(self):
        return AutoencoderMLPDatum(self._name, self.input, self.ground, self.output, to_numpy=True)

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
        datum.subplot_input(fig=fig, ax=ax)
        if path:
            plt.savefig(path)
        else:
            plt.show()
        plt.close(fig)
    
    @classmethod
    def load_data_from(cls, root):
        return (cls(filename=osp.join(root, dirs, dirs))
                for dirs in tqdm(os.listdir(root))
                if  osp.isfile(osp.join(root, dirs, dirs + '.img.npy' ))
                and osp.isfile(osp.join(root, dirs, dirs + '.u_cf.npy'))
                and osp.isfile(osp.join(root, dirs, dirs + '.l_cf.npy')))

    def __repr__(self):
        return '<ShallowDatum.AutoencoderMLP (name="{}")>'.format(self._name)

