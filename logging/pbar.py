from typing import overload
from .handler import TrainerLogger
from tqdm import tqdm
import time
import os


class ProgressBar(TrainerLogger):
    """
    A callback which visualises the state of each training and evaluation epoch using a progress bar
    """

    def __init__(self, delay=0.0):
        super(ProgressBar, self).__init__()
        self.pbar = None
        self._prog_size = None
        self._delay = delay
        self.varbose = False

    def log(self, trainer=None, **kwargs):
        self.pbar.set_postfix(**kwargs)

    def update(self, *args, **kwargs):
        self.pbar.update(1)

    def close(self):
        self.pbar.close()
        if self._delay > 0.0:
            time.sleep(self._delay)

    def on_training_begin(self, *args, batch_size=None, step_size=None, **kwargs):
        self._prog_size = step_size

    def on_training_epoch_begin(self, epoch=None, **kwargs):
        self.pbar = tqdm(
            total=self._prog_size,
            unit="batch",
            initial=0,
            file=os.sys.stdout,
            dynamic_ncols=True,
            desc=f"Epoch {epoch}",
            ascii=True,
            colour="GREEN",
        )

    def on_training_step_end(self,
                             trainer,
                             epoch=None,
                             batch=None,
                             batch_output=None,
                             **stepped_values):
        self.log(**{key: value.item()
                    for key, value in stepped_values.items()})
        self.update()

    def on_training_epoch_end(self, trainer, epoch=None, **updated_values):
        self.log(**{key: value.item()
                    for key, value in updated_values.items()})
        self.close()

    def on_validation_run_begin(self, batch_size=None, step_size=None, **kwargs):
        self.pbar = tqdm(
            total=step_size,
            unit="case",
            file=os.sys.stdout,
            dynamic_ncols=True,
            desc=f"Test",
            colour="GREEN",
            # postfix=self.metrics.keys(),
        )

    def on_validation_step_end(self,
                               trainer,
                               epoch=None,
                               batch=None,
                               batch_output=None,
                               **metric_values):
        metric_values = {key: value.item()
                         for key, value in metric_values.items()}
        self.log(**metric_values)
        self.update()

    def on_validation_run_end(self, trainer, **kwargs):
        self.close()
