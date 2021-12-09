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
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
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
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                trainer.stop_iter(restart=False)
                trainer.save_checkpoint(epoch=epoch)
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop


class SaveBestModel(Callback):
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_best_metric='valid_loss', verbose=False, this_max=False):
        self.save_best_metric = save_best_metric
        self.best_weights = None
        self.max = this_max
        self.verbose = verbose
        self.best = float('-inf') if this_max else flot('inf')

    def on_epoch_end(self, trainer, epoch=None, **kwargs):
        metric_value = kwargs[self.save_best_metric]

        if self.max and metric_value > self.best or metric_value < self.best:
            self.best = metric_value
            self.best_weights = trainer.model.state_dict()
            if self.verbose: 
                print("best model is saved...")

    def on_train_end(self, trainer, **kwargs):
        if self.best_weights is not None:
            self.model.save_state_dict(self.best_weights)
