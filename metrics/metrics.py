import torch
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Mapping, Optional, Union, Callable, Tuple, Iterable
from .handler.events import on_initialization, on_run_begin, on_run_end, on_step_begin, on_step_end


class __TrainerMetric(ABC):

    def __init__(self,
                 metrics: Mapping[str, Callable[[
                     torch.Tensor, torch.Tensor], float]] = dict(),
                 loss_list=None,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None
                 ):
        self.metrics = metrics
        self.loss_list = loss_list
        self.device = device
        self.dtype = dtype

    # @on_run_begin
    def init(self,
             batch_size: int,
             **kwargs,
             ) -> None:
        if self.dtype is not None:
            kwargs.setdefault('dtype', self.dtype)
        if self.device is not None:
            kwargs.setdefault('device', self.device)

        self.loss_list = torch.empty(
            batch_size, self.metrics.__len__(), **kwargs,
        )

    # @on_epoch_begin
    def reset(self):
        self.loss_list.zero_()

    # @on_epoch_end
    def update(self, epoch: int = None, *args, **kwargs):
        pass

    # @on_step_end
    def step(
        self,
        batch_idx: int,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        *args,
        **kwargs,
    ) -> None:
        self.loss_list[batch_idx, :] = torch.tensor(
            [
                self.metrics[metric_name](y_true, y_pred)
                for metric_name in self.metrics.keys()
            ]
        )

    def stepped_values(self, batch_idx: int) -> Mapping[str, torch.tensor]:
        return dict(zip(self.metrics.keys(), self.loss_list[batch_idx, :]))

    def updated_values(self, epoch: int, prefix: Optional[str] = None) -> Mapping[str, torch.Tensor]:
        return dict(zip(self.metrics.keys(), self.loss_list.mean(0)))

    def save(self):
        with open(self.path, "w") as f:
            f.write("\n".join(self.logs))


def TrainerMetric(*args, **kwargs):
    metric = __TrainerMetric(*args, **kwargs)
    on_initialization(metric.init)
    on_run_begin(metric.reset)
    on_run_end(metric.update)
    # on_step_begin()
    on_step_end(metric.step)
    return metric
