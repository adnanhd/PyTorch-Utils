#!/usr/bin/env python3
from .dataset import Dataset

def l1_norm(arr):
    nsum = arr.sum()
    arr /= nsum
    return nsum

def l1_denorm(arr, key):
    arr *= key
    

def mean_std_norm(arr):
    nmean = arr.mean()
    nstd = arr.std()
    
    arr -= nmean
    arr /= nstd
    
    return nmean, nstd

def mean_std_denorm(arr, key):
    nmean, nstd = key
    
    arr *= nstd
    arr += nmean
    

def min_max_norm(arr):
    nmin = arr.min()
    nmax = arr.max()
    
    arr -= nmin
    arr /= nmax - nmin
    
    return nmin, nmax

def min_max_denorm(arr, key):
    nmin, nmax = key
    
    arr *= nmax - nmin
    arr += nmin

