#!/usr/bin/env python3
import pandas as pd
import os, time
import torch, math
from tqdm import tqdm, trange, utils
from .utils import profile
from typing import (
    List, 
    Dict, 
    Any, 
    Mapping, 
    Optional, 
    Union, 
    Callable, 
    Tuple, 
    Iterable
)

def _run_validating(
    self,
    eval_loader: torch.utils.data.DataLoader,
) -> torch.Tensor:
    #metrics.reset(trainer=self)
    trainer.__handle__("on_valid_epoch_begin", self)

    self.model.eval()
    with torch.no_grad():
        for batch, (features, y_true) in enumerate(eval_loader):
            _run_evaluating_step(
                batch_idx=batch,
                x=features.to(device=self.device, dtype=self.xtype),
                y=y_true.to(device=self.device, dtype=self.ytype),
            )

    #metrics.update(epoch=index)
    trainer.__handle__("on_valid_epoch_end", epoch=batch)

def _run_validating_step(
    trainer,
    batch_idx: int,
    x: torch.Tensor,
    y: torch.Tensor,
    **kwargs,
):
    trainer.__handle__("on_eval_step_begin", step=batch_idx)

    y_pred = self.model(x)
    loss = self.criterion(y_pred, y)

    trainer.__handle__("on_eval_step_end", step=batch_idx)
