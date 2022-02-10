from ..callbacks import TrainerCallback
from abc import ABC, abstractmethod


class TrainerLogger(TrainerCallback, ABC):
    """
    A callback which visualises the state of each training and evaluation epoch using a progress bar
    """

    def __init__(self):
        self(TrainerLogger, self).__init__()

    @abstractmethod
    def log(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplementedError()
