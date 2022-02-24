# Copyright © 2021 Chris Hughes
import inspect
import logging
import sys, os
import time
from abc import ABC

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from .base import TrainerCallback


class ModelCheckpoint(TrainerCallback):
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, monitor="valid_loss", verbose=False, this_max=False):
        self.monitor = monitor
        self.best_weights = None
        self.max = this_max
        self.verbose = verbose
        self.best = float("-inf") if this_max else float("inf")

    def on_train_epoch_end(self, trainer, **kwargs):
        metric_value = kwargs[self.monitor]

        if self.max and metric_value > self.best or metric_value < self.best:
            self.best = metric_value
            self.best_weights = trainer.model.state_dict()
            if self.verbose:
                print("best model is saved...")

    def on_training_run_end(self, trainer, **kwargs):
        if self.best_weights is not None:
            self.model.save_state_dict(self.best_weights)


