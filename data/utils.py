import os
from abc import ABC, abstractmethod
import torch
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt


class Constants(object):
    preloaded_dataset_path = os.path.join(os.path.split(__file__)[0], '.PreLoadedDataset/DATA/')


def encoder_lambda(x_scale, y_min, y_max):
    """
    @brief generates a lambda function that converts a number 
           between @param y_min and @param y_max into a number 
           between 0 and @param x_scale

    @param x_scale the maximum number that encoded output can be (min. is 0)
    @param y_min the minimum input number that can cause any change in the output
    @param y_max the maximum input number that can cause any change in the output
    @return a lambda function which encodes its input
    """
    scale = x_scale / (y_max - y_min)
    shift = - scale * y_min
    return lambda y: max(0, min(x_scale, int(y * scale + shift)))


# bezier_curve = lambda x_points, degree: bc(x_points, degree, Utils.Point.linspace)
def generate_dataset(*datum_classes, filepath=None):
    if not filepath:
        filepath = Constants.preloaded_dataset_path
    
    contains = lambda parent, suffix: any(child.endswith(suffix) 
            for child in os.listdir(os.path.join(filepath, parent)))
    
    for datum_folder in os.listdir(filepath):
        if all(map(lambda datum_class: contains(datum_folder, datum_class.suffix), datum_classes)):
            yield datum_folder


