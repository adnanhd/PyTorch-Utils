#!/usr/bin/envy python3.6

import os

import numpy
import torch
import scipy.io
import numpy as np
import os.path as osp
from tqdm import tqdm
from torch.utils.data.dataset import Dataset

class MatLoader(Dataset):
    def __init__(self, data_list, mode=None, path=None):
        self.__data_list = list(data_list)
        self.__mode = mode
        self.__path = path

    def __getitem__(self, index):
        datum = self.__data_list[index]
        # will return DF,Y ,i.e, input-output
        return datum.path, datum.input, datum.ground

    def __len__(self):
        return len(self.__data_list)

    @classmethod
    def get_loader(cls, mode, batch_size=1, num_workers=None):
        raise NotImplementedError()

    def __repr__(self):
        return '<MatLoader (mode="{}", path="{}", size={})'.format(self.__mode, self.__path, len(self.__data_list))

