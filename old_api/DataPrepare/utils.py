import numpy as np
import torch

class Utils:
    def __init__(self, y_min=-0.256, y_max=.256, img_dim=216, head_margin=26, tail_margin=10, y_num=70, y_part=.4):
        self.y_min = y_min
        self.y_max = y_max
        self.img_dim = img_dim
        self.head_margin = head_margin
        self.tail_margin = tail_margin
        _slice = (y_num // 7) * 4
        self.x_fixed = np.linspace(0, 1, img_dim - head_margin - tail_margin + 1)
        self.y_fixed = np.linspace(y_min, y_max, img_dim + 1)
        self.img_encoder = Utils.encoder_lambda(img_dim - 1, y_min, y_max)
        self.x_sigmoid = np.append(np.linspace(0.0, y_part, _slice + 1)[
                                   :_slice], np.linspace(y_part, 1, y_num - _slice))
        self.is_debug = False
        self.x_split1000 = np.linspace(0, 1, 1000)

    @staticmethod
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

    def set_debug(self):
        self.is_debug = True

    def reset_debug(self):
        self.is_debug = False

    @staticmethod
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

