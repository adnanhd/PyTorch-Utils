#!/usr/bin/env python3

import os
import torch
import random
import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm
from .sample import StoredSample
#from airfoils import Airfoil
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import warnings
warnings.warn(f'<{__name__}> module is depricated in future versions', FutureWarning)


class CrudeDatum:
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
        return CrudeDatum.__load_airfoil(x, y)

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
        x_fixed = Image.usespace \
                  + self.airfoil.all_points[0][0]
        encoder, img_dim = Image.encoder, Image.shape[1]
        smargin, emargin = Image.head_margin, Image.tail_margin
        y_fixed = Image.linspace

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
        x_sigmoid = Point.linspace \
                    + self.airfoil.all_points[0][0]
        upper = self.airfoil.y_upper(x_sigmoid).reshape(Point.shape)
        lower = self.airfoil.y_lower(x_sigmoid).reshape(Point.shape)
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
        return interp1d(df['Position:0'].to_numpy(), df['Cf_upper'].to_numpy(), fill_value="extrapolate", bounds_error=False)(SkinFriction.linspace)

    @property
    def l_cf(self):
        df = pd.read_csv(os.path.join(self.cf_path, self.file_path, self.file_path + '_Cf_lower.csv')).drop_duplicates(subset=['Position:0'])
        return interp1d(df['Position:0'].to_numpy(), df['Cf_lower'].to_numpy(), fill_value="extrapolate", bounds_error=False)(SkinFriction.linspace)

    def save(self, dir_path):
        try:
            os.mkdir(dir_path)
        except FileExistsError:
            pass

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

        return StoredSample(file_prefix)

    @classmethod
    def load(cls, fname, coord_dir, bezier_dir, cf_dir):
        airfoil_name = fname.split('.')[0] + '_AOA=0'
        datum = cls(airfoil_name, coord_dir, bezier_dir, cf_dir)

        if 0.32 < datum.get_distribution() < 0.68:
            return datum
        else:
            raise Exception("partition error: {:03.1f}% in {}"
                    .format(p * 100.0, os.path.join(coord_dir, fname)))

    def get_distribution(self):
        return self.distribution

    def __repr__(self):
        return '<CrudeDatum (name="{}")>'.format(self.name)


def save_data_into(data_dir='../NewData2', 
        coord_dir='../Preprocessing/231/', 
        bezier_dir='../Preprocessing/BZR_DATA/bezier_coeffs_new/',
        cf_dir='../Preprocessing/CF_DATA/',
        verbose=False):
    try:
        os.mkdir(data_dir)
    except FileExistsError:
        pass

    for coord_file in tqdm(os.listdir(coord_dir)):
        try:
            datum = CrudeDatum.load(coord_file,
                    coord_dir=coord_dir,
                    bezier_dir=bezier_dir,
                    cf_dir=cf_dir)
            airfoil_name = coord_file.split('.')[0] + '_AOA=0'
            datum.save(os.path.join(data_dir, airfoil_name))
            yield datum

        except UnicodeDecodeError:
            if verbose:
                print("UnicodeDecodeError in {}".format(
                    os.path.join(coord_dir, coord_file)))
        except ValueError:
            if verbose:
                print("ValueError: in {}".format(coord_file))
        except FileExistsError:
            if verbose:
                print("FileExistsError: in {}".format(coord_file))
        except FileNotFoundError:
            if verbose:
                print("FileNotFoundError: in {}".format(coord_file))
        except RuntimeWarning:
            if verbose:
                print("RuntimeWarning: in {}".format(coord_file))
        except IndexError:
            if verbose:
                print("Index error {}".format(cf_dir + coord_file))

