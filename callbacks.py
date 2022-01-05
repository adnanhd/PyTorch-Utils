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


class StopTrainingError(Exception):
    """
    An exception which can be raised in order to stop a training run early.
    """

    pass


class CallbackMethodNotImplementedError(Exception):
    pass


class TrainerCallback(ABC):
    """
    The abstract base class to be subclassed when creating new callbacks.
    """

    def on_init_end(self, trainer, **kwargs):
        """
        Event called at the end of trainer initialisation.
        """
        pass

    def on_training_run_start(self, trainer, **kwargs):
        """
        Event called at the start of training run.
        """
        pass

    def on_train_epoch_start(self, trainer, **kwargs):
        """
        Event called at the beginning of a training epoch.
        """
        pass

    def on_train_step_start(self, trainer, **kwargs):
        """
        Event called at the beginning of a training step.
        """
        pass

    def on_train_step_end(self, trainer, batch, batch_output, **kwargs):
        """
        Event called at the end of a training step.
        :param batch: the current batch of training data
        :param batch_output: the outputs returned by :meth:`pytorch_accelerated.trainer.Trainer.calculate_train_batch_loss`
        """
        pass

    def on_train_epoch_end(self, trainer, **kwargs):
        """
        Event called at the end of a training epoch.
        """
        pass

    def on_eval_epoch_start(self, trainer, **kwargs):
        """
        Event called at the beginning of an evaluation epoch.
        """
        pass

    def on_eval_step_start(self, trainer, **kwargs):
        """
        Event called at the beginning of a evaluation step.
        """
        pass

    def on_eval_step_end(self, trainer, batch, batch_output, **kwargs):
        """
        Event called at the end of an evaluation step.
        :param batch: the current batch of evaluation data
        :param batch_output: the outputs returned by :meth:`pytorch_accelerated.trainer.Trainer.calculate_eval_batch_loss`
        """
        pass

    def on_eval_epoch_end(self, trainer, **kwargs):
        """
        Event called at the end of evaluation.
        """
        pass

    def on_training_run_epoch_end(self, trainer, **kwargs):
        """
        Event called during a training run after both training and evaluation epochs have been completed.
        """
        pass

    def on_training_run_end(self, trainer, **kwargs):
        """
        Event called at the end of training run.
        """
        pass

    def on_evaluation_run_start(self, trainer, **kwargs):
        """
        Event called at the start of an evaluation run.
        """
        pass

    def on_evaluation_run_end(self, trainer, **kwargs):
        """
        Event called at the end of an evaluation run.
        """
        pass

    def on_stop_training_error(self, trainer, **kwargs):
        """
        Event called when a stop training error is raised
        """
        pass

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            raise CallbackMethodNotImplementedError


class CallbackHandler:
    """
    The :class:`CallbackHandler` is responsible for calling a list of callbacks.
    This class calls the callbacks in the order that they are given.
    """

    def __init__(self, callbacks):
        self.callbacks = []
        self.add_callbacks(callbacks)

    def add_callbacks(self, callbacks):
        """
        Add a list of callbacks to the callback handler
        :param callbacks: a list of :class:`TrainerCallback`
        """
        for cb in callbacks:
            self.add_callback(cb)

    def add_callback(self, callback):
        """
        Add a callbacks to the callback handler
        :param callback: an instance of a subclass of :class:`TrainerCallback`
        """
        cb = callback() if isinstance(callback, type) else callback
        cb_class = callback if isinstance(callback, type) else callback.__class__
        if cb_class in {c.__class__ for c in self.callbacks}:
            raise ValueError(
                f"You attempted to add multiple instances of the callback {cb_class} to a single Trainer"
                f" The list of callbacks already present is\n: {self.callback_list}"
            )
        self.callbacks.append(cb)

    def __iter__(self):
        return self.callbacks

    def clear_callbacks(self):
        self.callbacks = []

    @property
    def callback_list(self):
        return "\n".join(cb.__class__.__name__ for cb in self.callbacks)

    def call_event(self, event, *args, **kwargs):
        """
        For each callback which has been registered, sequentially call the method corresponding to the
        given event.
        :param event: The event corresponding to the method to call on each callback
        :param args: a list of arguments to be passed to each callback
        :param kwargs: a list of keyword arguments to be passed to each callback
        """
        for callback in self.callbacks:
            try:
                getattr(callback, event)(
                    *args,
                    **kwargs,
                )
            except CallbackMethodNotImplementedError as e:
                continue


class EarlyStopping(TrainerCallback):
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience=7, verbose=False, delta=0, trace_func=print, save_model=True
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save_model = save_model
        self.val_loss = np.Inf
        self.delta = delta
        self.trace_func = trace_func

    def on_train_epoch_end(self, trainer, valid_loss=None, epoch=None, **kwargs):
        score = -valid_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
                trainer.stop_iter(restart=False)
                if self.save_model:
                    trainer.save_checkpoint(epoch=epoch)
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop


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


class ProgressBar(TrainerCallback):
    """
    A callback which visualises the state of each training and evaluation epoch using a progress bar
    """

    def __init__(self):
        self.pbar = None

    def on_train_epoch_start(self, trainer, epoch, **kwargs):
        self.pbar = tqdm(
            total=trainer._train_dataloader.__len__(),
            #disable=not trainer._accelerator.is_local_main_process,
            unit="batch",
            initial=0,
            file=os.sys.stdout,
            dynamic_ncols=True,
            desc=f"Epoch:{epoch}",
            ascii=True,
            colour="GREEN",
        )

    def on_eval_epoch_start(self, trainer, epoch, **kwargs):
        self.pbar = tqdm(
                trainer._eval_dataloader.__len__(),
                unit="case",
                file=os.sys.stdout,
                dynamic_ncols=True,
                desc=f"Test:{epoch}",
                colour="GREEN",
                postfix=metrics,
        )

    def on_train_step_end(self, trainer, **kwargs):
        self.pbar.update(1)

    def on_train_epoch_end(self, trainer, **kwargs):
        self.pbar.close()
        time.sleep(0.01)

    def on_eval_epoch_start(self, trainer, **kwargs):
        self.pbar = tqdm(
            total=len(trainer._eval_dataloader),
            disable=not trainer._accelerator.is_local_main_process,
        )

    def on_eval_step_end(self, trainer, **kwargs):
        self.pbar.update(1)
        self.pbar.set_postfix(
            **dict(zip(train_df.columns, loss_list[batch].cpu().tolist()))
        )

    def on_eval_epoch_end(self, trainer, **kwargs):
        self.pbar.close()
        time.sleep(0.01)
        if verbose:
            df = pd.DataFrame(
                (train_df.iloc[epoch], valid_df.iloc[epoch]),
                columns=metrics.keys(),
                index=["train", "valid"],
            )
            print(df)
