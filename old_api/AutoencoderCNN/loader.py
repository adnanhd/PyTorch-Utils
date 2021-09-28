#!/usr/bin/envy python3.6

import os
import torch
import numpy as np
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt

from DataPrepare import Utils, ShallowDatum, Datum, MatLoader

class AutoencoderCNNDatum(Datum):
    def __init__(self, title, inp, gnd, out=None, to_numpy=False):
        self.device = inp.device
        if to_numpy:
            inp = self.to_cpu(inp)
            inp.resize(Utils.Image.shape)

            gnd = gnd.reshape(Utils.Point.shape)
            gnd = np.concatenate((Utils.Point.linspace, self.to_cpu(gnd).reshape(1, -1)))
            #gnd.resize(2, 70)

            if not isinstance(None, type(out)):
                out = np.append(x_fixed, self.to_cpu(out))
                out.resize(2, 70)

        else:
            if not isinstance(None, type(out)):
                out = out.reshape(2, 70)
            gnd = gnd.reshape(2, 70)
            inp = inp.reshape(Utils.Image.shape)

        super(AutoencoderCNNDatum, self).__init__(title, _input=inp, _ground=gnd, _output=out)

    def subplot_input(self, ax=None, fig=None):
        assert (fig == None) == (ax == None), """Fig cannot be none unless ax is none."""
        if not isinstance(self.inp, torch.Tensor):
            image = torch.from_numpy(self.inp)
        else:
            image = self.inp.detach().cpu()
        mask = torch.ones(Utils.Image.shape)
        masked = torch.logical_and(mask, image)

        if not ax:
            fig, ax = plt.subplots()
        ax.set_title('Input Image of the Airfoil')
        conf = ax.contourf(masked, cmap=plt.cm.jet)
        fig.colorbar(conf, ax=ax)

    def subplot_ground(self, ax=None, u_color='red', l_color='orange'):
        if isinstance(self.gnd, torch.Tensor):
            Utils.Point.linspace = torch.from_numpy(x_fixed)
            pts = self.to_cpu(self.gnd)
        else:
            Utils.Point.linspace = x_fixed
            pts = self.gnd

        if not ax:
            fig, ax = plt.subplots()
        ax.scatter(Utils.Point.linspace, pts[0], facecolors='none', edgecolors=u_color, label='ground upper airfoil')
        ax.scatter(Utils.Point.linspace, pts[1], facecolors='none', edgecolors=l_color, label='ground lower airfoil')
        ax.set_ylim(y_lower_lim, y_upper_lim)  # +- 0.2
        ax.legend()

    def subplot_output(self, ax=None, u_color='blue', l_color='cyan'):
        if isinstance(self.out, torch.Tensor):
            Utils.Point.linspace = torch.from_numpy(x_fixed)
            pts = self.to_cpu(self.out)
        else:
            Utils.Point.linspace = x_fixed
            pts = self.out

        if not ax:
            fig, ax = plt.subplots()
        ax.plot(Utils.Point.linspace, pts[0], 'o-', color=u_color, label='output upper airfoil')
        ax.plot(Utils.Point.linspace, pts[1], 'o-', color=l_color, label='output lower airfoil')
        ax.set_ylim(Utils.Image.y_min, Utils.Image.y_max)  # +- 0.2
        ax.legend()

    def plot(self, path=None, title=None):
        fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 12))

        if not title:
            title = osp.split(self._name)[1]
        fig.suptitle(title)
        self.subplot_input(ax=ax1, fig=fig)
        ax2.set_title('Output Points of the Airfoil')

        if not isinstance(None, type(self.out)):
            self.subplot_output(ax=ax2)
        
        self.subplot_ground(ax=ax2)

        if path:
            plt.savefig(path)
        else:
            plt.show()
        plt.close(fig)

    def __repr__(self):
        return '<Datum.CNN (name="{}", device="{}")>'.format(self._name, self.device)


class AutoencoderCNNShallowDatum(ShallowDatum):
    def __init__(self, filename, output=None):
        super(AutoencoderCNNShallowDatum, self).__init__(filename)
        self.output = output

    @property
    def input(self):
        return torch.from_numpy(super()._img).unsqueeze(dim=0)

    @property
    def ground(self):
        return torch.from_numpy(self._pts).flatten()

    @classmethod
    def load_data_from(cls, root):
        return (cls(filename=osp.join(root, dirs, dirs))
                for dirs in tqdm(os.listdir(root))
                if osp.isfile(osp.join(root, dirs, dirs + '.img.npy'))
                and osp.isfile(osp.join(root, dirs, dirs + '.pts.npy')))

    @property
    def container(self):
        return AutoencoderCNNDatum(title=self._name, inp=self.input, gnd=self.ground, out=self.output, to_numpy=False)

    def plot(self, title=None, path=None):
        datum = self.container
        datum.plot(title)
        if path:
            plt.savefig(path)
        else:
            plt.show()
        plt.close()

    """
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
    """

    def __repr__(self):
        return '<ShallowDatum.CNN (name="{}")>'.format(self._name)


if __name__ == '__main__':
    loader = MatLoader('train', '../NewData/')
    data_list = AutoencoderCNNShallowDatum.load_data_from('../NewData/TRAIN/')

