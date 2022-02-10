#!/usr/bin/env python3
from .dataset import Dataset

def l1_norm(arr, dim=None, shape=-1):
    nsum = arr.sum(dim)
    arr /= nsum.reshape(shape)
    return nsum

def l1_denorm(arr, key, shape=-1):
    arr *= key.reshape(shape)
    

def mean_std_norm(arr, dim=None, shape=-1):
    nmean = arr.mean(dim)
    nstd = arr.std(dim)
    
    arr -= nmean.reshape(shape)
    arr /= nstd.reshape(shape)
    
    return nmean, nstd

def mean_std_denorm(arr, key, shape=-1):
    nmean, nstd = key
    
    arr *= nstd.reshape(shape)
    arr += nmean.reshape(shape)
    

def min_max_norm(arr, dim=None, shape=-1):
    nmin = arr.min(dim)
    nmax = arr.max(dim)
    
    arr -= nmin.reshape(shape)
    arr /= (nmax - nmin).reshape(shape)
    
    return nmin, nmax

def min_max_denorm(arr, key, shape=-1):
    nmin, nmax = key
    
    arr *= (nmax - nmin).reshape(shape)
    arr += nmin.reshape(shape)

