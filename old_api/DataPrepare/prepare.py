#!/usr/bin/env python3

import os
import torch
import random
import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm
from .datum import __Datum as ShallowDatum
from .utils2 import Utils, encoder_lambda
from airfoils import Airfoil
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pdb


class PrepareDatum:
    def __init__(self, file_path, coord_path, bezier_path, cf_path):
        """! The GenerativeDatum class initialized.

        @param cpath The path to airfoil coordinates
        @param bpath The path to beziyer coordinates

        @return
        """
        self.name = file_path.split('_AOA')[0]
        self.file_path = file_path
        self.coord_path = coord_path
        self.bezier_path = bezier_path
        self.cf_path = cf_path        

        x_coord, y_coord = self.__load_coord(os.path.join(coord_path, self.name + '.dat'))
        self.airfoil, self.distribution = self.__load_airfoil(x_coord, y_coord)

    @staticmethod
    def load_airfoil(x, y):
        return PrepareDatum.__load_airfoil(x, y)

    @staticmethod
    def norm(p1, p2):
        return np.sqrt(np.sum(np.square(p1 - p2)))

    @staticmethod
    def distance(l_st, l_en, pt):
        return np.abs(np.cross(l_en - l_st, pt - l_st) / np.sqrt(np.sum(np.square(l_en - l_st))))

    @staticmethod
    def __load_coord(filepath):
        # first check if file is saved as npy file
        if '.npy' in filepath:
            return np.load(filepath)
        # otherwise assume it is a dat file
        else:
            with open(filepath) as f:
                pairs = [line.strip().split()
                         for line in f.readlines()
                         if line.count('.') == 2
                         and not line[0].isalpha()]

            # old version : pairs = [pair for pair in pairs if len(pair) == 2]
            pairs = filter(lambda pair: len(pair) == 2, pairs)

            coord = np.array([[float(x), float(y)]
                              for [x, y] in pairs], dtype=np.float64)

            return coord.transpose()

    @staticmethod
    def __load_airfoil(x_coord, y_coord):
        search = list(x_coord[2:-2])
        chn = search.index(min(search)) + 2
        upper = x_coord[:chn + 1], y_coord[:chn + 1]
        lower = x_coord[chn:], y_coord[chn:]

        return Airfoil(upper, lower), chn / len(x_coord)

    @property
    def img(self):  # Proloque of Saving
        x_fixed = Utils.Image.usespace \
                  + self.airfoil.all_points[0][0]
        encoder, img_dim = Utils.Image.encoder, Utils.Image.shape[1]
        smargin, emargin = Utils.Image.head_margin, Utils.Image.tail_margin
        y_fixed = Utils.Image.linspace

        upper = self.airfoil.y_upper(x_fixed)
        lower = self.airfoil.y_lower(x_fixed)

        _min_dist = 1e4
        _max_dist = np.sqrt(2) / 200
        img_float = np.zeros((1, img_dim, img_dim)) + _max_dist

        for i, col in enumerate(img_float[0][smargin:img_dim - emargin], start=0):
            p1 = np.array([x_fixed[i], upper[i]])
            p2 = np.array([x_fixed[i + 1], upper[i + 1]])

            start_pixel = min(encoder(upper[i]), encoder(upper[i + 1]))
            end_pixel = max(encoder(upper[i]), encoder(upper[i + 1]))

            for j in range(start_pixel, end_pixel + 1):
                p3 = np.array([np.mean(x_fixed[i:i + 2]),
                               np.mean(y_fixed[j:j + 2])])
                col[j] = self.distance(p1, p2, p3)
                _min_dist = min(_min_dist, col[j])
                _max_dist = max(_max_dist, col[j])

            p1 = np.array([x_fixed[i], lower[i]])
            p2 = np.array([x_fixed[i + 1], lower[i + 1]])

            start_pixel = min(encoder(lower[i]), encoder(lower[i + 1]))
            end_pixel = max(encoder(lower[i]), encoder(lower[i + 1]))

            for j in range(start_pixel, end_pixel + 1):
                p3 = np.array([np.mean(x_fixed[i:i + 2]),
                               np.mean(y_fixed[j:j + 2])])
                col[j] = self.distance(p1, p2, p3)
                _min_dist = min(_min_dist, col[j])
                _max_dist = max(_max_dist, col[j])

        # Epiloque of Saving
        float2uint8 = encoder_lambda(255, _max_dist, _min_dist)
        return np.array([[list(map(float2uint8, col))
                          for col in img_float.squeeze().transpose()]], dtype=np.uint8)

    @property
    def pts(self):
        x_sigmoid = Utils.Point.linspace \
                    + self.airfoil.all_points[0][0]
        upper = self.airfoil.y_upper(x_sigmoid).reshape(Utils.Point.shape)
        lower = self.airfoil.y_lower(x_sigmoid).reshape(Utils.Point.shape)
        return np.concatenate((upper, lower))

    @property
    def u_bzr(self):
        return self.__load_coord(os.path.join(self.bezier_path, self.name + '_upper.dat'))

    @property
    def l_bzr(self):
        return self.__load_coord(os.path.join(self.bezier_path, self.name + '_lower.dat'))

    @property
    def u_cf(self):
        df = pd.read_csv(os.path.join(self.cf_path, self.file_path, self.file_path + '_Cf_upper.csv')).drop_duplicates(subset=['Position:0'])
        return interp1d(df['Position:0'].to_numpy(), df['Cf_upper'].to_numpy(), fill_value="extrapolate", bounds_error=False)(Utils.SkinFriction.linspace)

    @property
    def l_cf(self):
        df = pd.read_csv(os.path.join(self.cf_path, self.file_path, self.file_path + '_Cf_lower.csv')).drop_duplicates(subset=['Position:0'])
        return interp1d(df['Position:0'].to_numpy(), df['Cf_lower'].to_numpy(), fill_value="extrapolate", bounds_error=False)(Utils.SkinFriction.linspace)

    def save_data(self, dir_path):
        try:
            os.mkdir(dir_path)
        except FileExistsError:
            pass
        # u_bzr, l_bzr  = self.u_bzr, self.l_bzr
        # image, points = self.img, self.pts
        # u_cf, l_cf    = self.u_cf, self.l_cf

        file_prefix = os.path.join(dir_path, self.file_path)
        
        np.save(file_prefix + '.pts',   self.pts)
        
        try:
            np.save(file_prefix + '.u_cf',  self.u_cf)
            np.save(file_prefix + '.l_cf',  self.l_cf)
        except FileNotFoundError:
            pass

        np.save(file_prefix + '.u_bzr', self.u_bzr)
        np.save(file_prefix + '.l_bzr', self.l_bzr)
        np.save(file_prefix + '.img',   self.img)

        return ShallowDatum(file_prefix)

    def get_distribution(self):
        return self.distribution

    def get_fig(self):
        pts = self.pts
        img = torch.from_numpy(self.img)
        u_bzr = self.u_bzr
        l_bzr = self.l_bzr
        mask = torch.ones(Utils.Image.shape)
        masked = torch.logical_and(mask, img.squeeze())
        fig = plt.figure(figsize=(11, 11))
        aximg = fig.add_subplot(2, 1, 1)
        plt.title(self.name)
        aximg.contourf(masked, cmap=plt.cm.jet)
        axpts = fig.add_subplot(2, 1, 2)
        axpts.scatter(Utils.Point.linspace, pts[0], color='red')
        axpts.scatter(Utils.Point.linspace, pts[1], color='red')
        axpts.scatter(u_bzr[0], u_bzr[1], color='blue')
        axpts.scatter(l_bzr[0], l_bzr[1], color='green')
        plt.xlim(-0.1, 1.1)
        plt.ylim(Utils.Image.y_min, Utils.Image.y_max)
        return fig

    @classmethod
    def prepare_data_from(cls, coord_dir, bezier_dir, data_dir, cf_dir, train=0.8, test=0.5):
        error_list = []
        reference_data = []
        try:
            os.mkdir(data_dir)
        except FileExistsError:
            pass

        for mode in ['TRAIN', 'TEST', 'VALID']:
            try:
                os.mkdir(os.path.join(data_dir, mode))
            except FileExistsError:
                pass

        for coord_file in tqdm(os.listdir(coord_dir)):
            if random.random() < train:
                mode = "TRAIN"
            elif random.random() < test:
                mode = "TEST"
            else:
                mode = "VALID"

            try:
                airfoil_name = coord_file.split('.')[0] + '_AOA=0'
                datum = cls(airfoil_name, coord_dir, bezier_dir, cf_dir)
                p = datum.get_distribution()
                #pdb.set_trace()

                if 0.32 < p < 0.68:
                    reference_data.append(datum.save_data(os.path.join(data_dir, mode, airfoil_name)))
                else:
                    error_list.append("partition error: {}%% in {}".format(
                        round(p, 0), coord_file))

            except UnicodeDecodeError:
                error_list.append("UnicodeDecodeError in {}".format(
                    os.path.join(coord_dir, coord_file)))
            except ValueError:
                error_list.append("ValueError: in {}".format(coord_file))
            except FileExistsError:
                error_list.append("FileExistsError: in {}".format(coord_file))
            except FileNotFoundError:
                error_list.append("FileNotFoundError: in {}".format(coord_file))
            except RuntimeWarning:
                error_list.append("RuntimeWarning: in {}".format(coord_file))
            except IndexError:
                print("Index error {}".format(cf_dir + coord_file))

        if error_list:
            print('{} error occured:'.format(len(error_list)))
            for error in error_list:
                print(error)

        return reference_data

    def __repr__(self):
        return '<PrepareDatum (name="{}")>'.format(self.name)


def save_data_into(datapath='../Data'):
    return PrepareDatum.prepare_data_from(
        coord_dir='../Preprocessing/1550/',
        bezier_dir='../Preprocessing/BZR_DATA/bezier_coeffs_new/',
        data_dir=datapath,
        cf_dir="../Preprocessing/CF_DATA/"
    )
