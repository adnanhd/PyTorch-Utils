#!/usr/bin/env python3
from typing import Tuple
import numpy as np
import torch
import warnings
from .dataset import Dataset


def _obtain_shape(dim, arr_shape):
    shape = tuple(1 if d in dim else -1 for d in range(len(arr_shape)))
    if shape.count(-1) > 1:
        warnings.warn(
            f'Obtained more than one dimension in dim={dim} and arr_shape={arr_shape}', RuntimeWarning)
    elif shape.count(-1) == 0:
        warnings.warn(
            f'Obtained no dimension in dim={dim} and arr_shape={arr_shape}', RuntimeWarning)
    else:
        return shape


def _default_dim_check(arr, dim):
    if isinstance(arr, torch.Tensor):
        return tuple()
    else:
        return None


def normalizer(normalizer_fn):
    def wrapped_fn(arr, dim=None, shape=-1):
        if isinstance(dim, type(None)):
            dim = _default_dim_check(arr, dim)
        if shape == -1 and not isinstance(dim, type(None)):
            shape = _obtain_shape(dim, arr.shape)
            if shape is None:
                shape = -1
        return normalizer_fn(arr, dim, shape), shape
    return wrapped_fn


def denormalizer(denormalizer_fn):
    def wrapped_dfn(arr, key):
        _key, shape = key
        return denormalizer_fn(arr, _key, shape)
    return wrapped_dfn


@normalizer
def l1_norm(arr, dim=None, shape=-1) -> Tuple[int, int]:
    nsum = arr.sum(dim)
    arr /= nsum.reshape(shape)
    return nsum


@denormalizer
def l1_denorm(arr, key, shape=-1) -> None:
    arr *= key.reshape(shape)


@normalizer
def mean_std_norm(arr, dim=None, shape=-1) -> Tuple[int, int]:
    nmean = arr.mean(dim)
    nstd = arr.std(dim)

    arr -= nmean.reshape(shape)
    arr /= nstd.reshape(shape)

    return nmean, nstd


@denormalizer
def mean_std_denorm(arr, key, shape=-1) -> None:
    nmean, nstd = key

    arr *= nstd.reshape(shape)
    arr += nmean.reshape(shape)


@normalizer
def min_max_norm(arr, dim=None, shape=-1) -> Tuple[int, int]:
    nmin = arr.min(dim)
    nmax = arr.max(dim)

    arr -= nmin.reshape(shape)
    arr /= (nmax - nmin).reshape(shape)

    return nmin, nmax


@denormalizer
def min_max_denorm(arr, key, shape=-1) -> None:
    nmin, nmax = key

    arr *= (nmax - nmin).reshape(shape)
    arr += nmin.reshape(shape)
