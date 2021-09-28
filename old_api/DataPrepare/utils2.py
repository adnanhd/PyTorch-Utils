#!/usr/bin/env python3

import numpy as np
import torch

    
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


def bezier_curve(x_points, degree, x_input):
    """x = bezierCurve(bezierCoefficients,degree)
    Returns 1x10000 array of a Bernstein polynomial."""
    def mul(arr):
        if arr:
            return arr[0] * mul(arr[1:])
        else:
            return 1

    if isinstance(x_points, torch.Tensor):
        t = torch.from_numpy(inp_x).to(x_points.device)
        f = torch.zeros(t.shape.numel()).to(x_points.device)
    else:
        t = x_input
        f = np.zeros(mul(t.shape))
    for k in range(degree + 1):
        f = f + (np.math.factorial(degree) / (np.math.factorial(k) * np.math.factorial(degree - k))
                 * (1 - t) ** (degree - k) * t ** k) * x_points[k]

    return f


class Utils:
    debug = False
    
    class Image:
        y_min = -.256
        y_max = +.256 

        shape = (216, 216)
        head_margin = 26
        tail_margin  = 10

        # spacing into how many useful pixels are there in an image
        usespace = np.linspace(0, 1, shape[1] - head_margin, tail_margin + 1)
        
        encoder = encoder_lambda(shape[1] - 1, y_min, y_max)

        linspace = np.linspace(y_min, y_max, shape[1] + 1)

    class Point:
        shape = (1, 70)
        y_part= +.4
        __slice = (shape[1] // 7) * 4

        linspace = np.append(np.linspace(0.0, y_part, __slice + 1)[
                                   :__slice], np.linspace(y_part, 1, shape[1] - __slice))

    class Bezier:
        shape = (1, 2, 6)
        linspace = None

    class DistField:
        shape = None
        linspace = None

    class SkinFriction:
        shape = (1, 1000)
        linspace = np.linspace(0, 1, shape[1])


