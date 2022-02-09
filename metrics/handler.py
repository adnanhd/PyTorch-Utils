import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Mapping, Optional, Union, Callable, Tuple, Iterable
from .utils import on_run_begin, on_epoch_begin, on_epoch_end, on_step_end

class MetricHandler:
    def __init__(
        self,
        metrics: Mapping[str, Callable[[torch.Tensor, torch.Tensor], float]] = {},
    ):
        self.metrics = metrics
        self.loss_list = None
        #self._df = None

    @on_run_begin
    def init(self, 
            batch_size: int,
            **kwargs,
    ) -> None:
        self.loss_list = torch.empty(
            batch_size, self.metrics.__len__(),
            **kwargs, #dtype=dtype, device=device,
        )

    @on_epoch_begin
    def reset(self):
        self.loss_list.zero_()

    @on_epoch_end
    def update(self, epoch: int):
        pass 
    
    @on_step_end
    def step(
        self,
        batch_idx: int,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
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
