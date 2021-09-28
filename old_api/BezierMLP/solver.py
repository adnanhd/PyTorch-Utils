#!/usr/bin/envy python3.6

import os
import os.path as osp
import torch
import torch.nn as nn
import torch.optim as optimizer
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import numpy as np
from .model import BezierMLPModel as Model
from .loader import MLPDatum as Datum, inp_x


def bezier_curve(points, degree, y_dim=70):
    def __bezier_curve(x_points):
        """x = bezierCurve(bezierCoefficients,degree)
        Returns 1x10000 array of a Bernstein polynomial."""
        t = torch.from_numpy(inp_x).to(x_points.device)
        f = torch.zeros(t.shape.numel()).to(x_points.device)
        for k in range(degree + 1):
            f = f + (np.math.factorial(degree) / (np.math.factorial(k) * np.math.factorial(degree - k))
                     * (1 - t) ** (degree - k) * t ** k) * x_points[k]

        return f

    curves = torch.zeros(len(points), y_dim)
    for y, pts in enumerate(points):
        curves[y] = __bezier_curve(pts)

    return curves


def subplot_train(epoch_num, train_loss, valid_loss, title=None, ax=None, label_suffix=''):
    if not ax:
        fig, ax = plt.subplots()
    if title:
        ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('L2 Loss')
    ax.semilogy(range(epoch_num), train_loss, label='train ' + label_suffix)
    ax.semilogy(range(epoch_num), valid_loss, label='valid ' + label_suffix)
    ax.set_ylim(1e-4, 1e0)
    ax.legend()


def subplot_test(epoch_num, test_loss, ax=None):
    if not ax:
        fig, ax = plt.subplots()
    ax.hist(test_loss, bins=20)
    ax.set_title('Test Error Histogram by Pixel at {} Epoch'.format(epoch_num))
    ax.set_ylabel('Number Of Samples')
    ax.set_xlabel('Mean Square Error (avg. {:.3e} px)'.format(sum(test_loss) / len(test_loss)))


class Solver(object):
    def __init__(self, args, data_loader, valid_loader):

        self.mode = args.mode
        self.data_loader = data_loader
        self.valid_loader = valid_loader
        self.save_path = args.save_path
        self.num_epochs = args.NumEpochs
        self.start_epochs = args.StartEpochs
        self.lr = args.LearnRate
        self.weight_decay = args.weightDecay
        self.print_iters = args.print_iters
        self.save_iters = args.save_iters
        self.test_iters = args.test_iters
        self.restart_iters = args.restart_iters
        self.multi_gpu = args.multi_gpu

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.model = Model().to(self.device)
        self.optimizer = optimizer.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.loss_func = nn.MSELoss()
        self.loss_func.to(self.device)
        self.temp = None

    def img_to(self, df):
        return df.to(device=self.device, dtype=torch.float)

    def save_model(self, state):
        epoch = state['epoch']
        f = osp.join(self.save_path, 'CFD_CNN_{}iter.ckpt'.format(epoch))
        torch.save(state, f)
        torch.save(np.array(state['train_loss']), osp.join(self.save_path,
                                                           'train_loss_{}_iter.npy'.format(epoch)))
        torch.save(np.array(state['valid_loss']), osp.join(self.save_path,
                                                           'valid_loss{}_iter.npy'.format(epoch)))

    def load_model(self, epoch):
        f = osp.join(self.save_path, 'CFD_CNN_{}iter.ckpt'.format(epoch))
        checkpoint = torch.load(f, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return self.model, self.optimizer, checkpoint['epoch'], checkpoint['train_loss'], checkpoint['valid_loss']

    def train(self):
        train_loss_lst = []
        valid_loss_lst = []
        start_time = time.time()

        # RESTART LINES #
        # model, optimizer, resumeEpochs, train_loss, valid_loss = self.load_model(self.restart_iters)
        # self.start_epochs = resumeEpochs

        for epoch in tqdm(range(self.start_epochs, self.num_epochs)):

            self.model.train(True)
            loss_list = []
            for i, (filename, dist_func, y_coordinates) in enumerate(self.data_loader):
                dist_func = self.img_to(dist_func)
                y_coordinates = self.img_to(y_coordinates)

                y_prediction = self.model(dist_func)

                loss = self.loss_func(y_prediction, y_coordinates)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_list.append(loss / len(filename))
            train_loss_lst.append(sum(loss_list) / len(self.data_loader))

            self.model.eval()
            with torch.no_grad():  # VALIDATION
                loss_list = []
                for i, (filename, dist_func, y_coordinates) in enumerate(self.valid_loader):
                    dist_func = self.img_to(dist_func)
                    y_coordinates = self.img_to(y_coordinates)

                    y_output = self.model(dist_func)

                    loss = self.loss_func(y_output, y_coordinates)

                    loss_list.append(loss / len(filename))
                valid_loss_lst.append(sum(loss_list) / len(self.valid_loader))
                if (epoch + 1) % self.save_iters == 0:
                    self.save_model({
                        'epoch': epoch + 1,
                        'train_loss': train_loss_lst,
                        'valid_loss': valid_loss_lst,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict()
                    })

        fig, ax = plt.subplots()
        fig.suptitle('L2Error and Train Loss Per Epoch with Curve Diff')
        subplot_train(epoch_num=self.num_epochs,
                      train_loss=train_loss_lst,
                      valid_loss=valid_loss_lst,
                      label_suffix='point diff',
                      ax=ax)
        plt.savefig(osp.join(self.save_path, "train_loss_logy.png"))
        plt.close(fig)

        print("TIME: {:.1f}s ".format(time.time() - start_time))

    def train_extended(self):
        train_loss_lst = []
        valid_loss_lst = []
        start_time = time.time()

        # RESTART LINES #
        # model, optimizer, resumeEpochs, train_loss, valid_loss = self.load_model(self.restart_iters)
        # self.start_epochs = resumeEpochs

        for epoch in tqdm(range(self.start_epochs, self.num_epochs)):

            self.model.train(True)
            loss_list = []
            for i, (filename, dist_func, y_coordinates) in enumerate(self.data_loader):
                dist_func = self.img_to(dist_func)
                y_coordinates = self.img_to(y_coordinates)

                y_prediction = self.model(dist_func)

                y_out = bezier_curve(y_prediction, 5)
                y_gnd = bezier_curve(y_coordinates, 5)

                loss = self.loss_func(y_out, y_gnd)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_list.append(loss / len(filename))
            train_loss_lst.append(sum(loss_list) / len(self.data_loader))

            self.model.eval()
            with torch.no_grad():  # VALIDATION
                loss_list = []
                for i, (filename, dist_func, y_coordinates) in enumerate(self.valid_loader):
                    dist_func = self.img_to(dist_func)
                    y_coordinates = self.img_to(y_coordinates)

                    y_output = self.model(dist_func)

                    y_out = bezier_curve(y_output, 5)
                    y_gnd = bezier_curve(y_coordinates, 5)

                    loss = self.loss_func(y_out, y_gnd)

                    loss_list.append(loss / len(filename))
                valid_loss_lst.append(sum(loss_list) / len(self.valid_loader))

                if (epoch + 1) % self.save_iters == 0:
                    self.save_model({
                        'epoch': epoch + 1,
                        'train_loss': train_loss_lst,
                        'valid_loss': valid_loss_lst,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict()
                    })

        fig, ax = plt.subplots()
        fig.suptitle('L2Error and Train Loss Per Epoch with Curve Diff')
        subplot_train(epoch_num=self.num_epochs,
                      train_loss=train_loss_lst,
                      valid_loss=valid_loss_lst,
                      label_suffix='curve diff',
                      ax=ax)
        plt.savefig(osp.join(self.save_path, "train_loss_logy.png"))
        plt.close(fig)

        print("TIME: {:.1f}s ".format(time.time() - start_time))

    def test(self):
        mse_pixel = []
        mse_percent = []
        start_time = time.time()

        self.load_model(self.test_iters)
        self.model.eval()

        try:
            os.mkdir(osp.join(self.save_path, "test"))
        except FileExistsError:
            print('test file exists under ' + self.save_path)

        with torch.no_grad():
            for i, (filename, dist_func, y_coordinate) in enumerate(tqdm(self.data_loader)):
                dist_func = self.img_to(dist_func)
                y_coordinate = self.img_to(y_coordinate)
                y_prediction = self.model(dist_func)

                datum = Datum(
                    title=filename[0],
                    inp=dist_func[0],
                    out=y_prediction[0],
                    gnd=y_coordinate[0]
                )

                mse_pixel.append(datum.mse)

                datum.plot(path=osp.join(self.save_path, 'test', osp.split(filename[0])[1]))

        fig, ax = plt.subplots()
        subplot_test(epoch_num=self.test_iters, test_loss=mse_pixel, ax=ax)
        plt.savefig(osp.join(self.save_path, 'test_loss_hist_px_{}.png'.format(self.test_iters)))
        # plt.show()
        plt.close(fig)

        print("TIME: {:.1f}s Error:{:.2f}px ".format(time.time() - start_time,
                                                     sum(mse_pixel) / len(mse_pixel)))
