import numpy as np
import torch


class Callback:
    def __init__(self):
        pass

    def on_training_end(self, trainer, **kwargs):
        pass

    def on_testing_end(self, trainer, **kwargs):
        pass

    def on_epoch_end(self, trainer, **kwargs):
        pass


class EarlyStopping(Callback):
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print, save_model=True):
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

    def on_epoch_end(self, trainer, valid_loss=None, epoch=None, **kwargs):
        score = -valid_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                trainer.stop_iter(restart=False)
                if self.save_model:
                    trainer.save_checkpoint(epoch=epoch)
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop


class ModelCheckpoint(Callback):
    from copy import deepcopy
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, monitor='valid_loss', verbose=False, save_max=False, save_model=False, step_size=10):
        self.monitor = monitor
        self.best_weights = None
        self.max = save_max
        self.save_model = save_model
        self.verbose = verbose
        self.step_size = step_size
        self.best = float('-inf') if save_max else float('inf')

    def on_epoch_end(self, trainer, epoch=None, **kwargs):
        metric_value = kwargs[self.monitor]

        if epoch % self.step_size == 0:
            if self.max and metric_value > self.best or metric_value < self.best:
                self.best = metric_value
                self.best_weights = (trainer.model.state_dict())   # parantez dışı self.deepcopy
                if self.verbose:
                    print("best model is saved...")

    def on_training_end(self, trainer,epoch, **kwargs):
        if self.best_weights is not None:
            print(type(self.best_weights))
            trainer.model.load_state_dict(self.best_weights)
            if self.verbose:
                print("best model is loaded back to model...")

        if self.save_model:
            trainer.save_checkpoint(epoch=epoch)
            if self.verbose:
                print("best model is saved to a path...")
