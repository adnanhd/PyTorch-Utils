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

def _add_prefix(prefix: str, kwargs: Mapping[str, Any]):
    return {f'{prefix}_{key}': value for key, value in kwargs.items()}

@profile
def _run_training(
    trainer,
    num_epochs: int,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: Optional[torch.utils.data.DataLoader] = None,
    **kwargs,
):
    _UNROLLING_N = 8
    trainer.callbacks.call_event("on_training_run_begin")

    print(num_epochs, num_epochs - _UNROLLING_N)
    for epoch in range(0, num_epochs - _UNROLLING_N + 1, _UNROLLING_N):
        _run_training_epoch(trainer, epoch+0, train_loader, valid_loader, **kwargs)
        _run_training_epoch(trainer, epoch+1, train_loader, valid_loader, **kwargs)
        _run_training_epoch(trainer, epoch+2, train_loader, valid_loader, **kwargs)
        _run_training_epoch(trainer, epoch+3, train_loader, valid_loader, **kwargs)
        _run_training_epoch(trainer, epoch+4, train_loader, valid_loader, **kwargs)
        _run_training_epoch(trainer, epoch+5, train_loader, valid_loader, **kwargs)
        _run_training_epoch(trainer, epoch+6, train_loader, valid_loader, **kwargs)
        _run_training_epoch(trainer, epoch+7, train_loader, valid_loader, **kwargs)

    for epoch in range((num_epochs // _UNROLLING_N) * _UNROLLING_N, num_epochs):
        _run_training_epoch(trainer, epoch, train_loader, valid_loader, **kwargs)

    trainer.callbacks.call_event("on_training_run_end", epoch=num_epochs)

@profile
def _run_training_epoch(
    trainer,
    epoch: int,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: Optional[torch.utils.data.DataLoader] = None,
    **kwargs,
) -> torch.Tensor:
    trainer.metrics.reset()
    trainer.callbacks.call_event("on_train_epoch_begin", epoch=epoch+1)

    trainer.model.train()
    for batch, (features, y_truth) in enumerate(train_loader):
        _run_training_step(
            trainer=trainer, epoch=epoch, batch_idx=batch, **kwargs,
            x=features.to(device=trainer.device, dtype=trainer.xtype),
            y=y_truth.to(device=trainer.device, dtype=trainer.ytype),
        )
    
    trainer.metrics.update(epoch)
    #train_values_dict = trainer.metrics.updated_values(epoch)
    #train.loggers.update(epoch, **_add_prefix('train', train_values_dict))
    trainer.callbacks.call_event("on_train_epoch_end", epoch=epoch+1, 
            **trainer.metrics.updated_values(epoch))
    
    if valid_loader is not None:
        _run_evaluate(
            trainer,
            valid_loader,
            **kwargs,
        )
        
    trainer.callbacks.call_event("on_train_eval_epoch_end", epoch=epoch+1)

#@profile
def _run_training_step(
    trainer,
    epoch: int,
    batch_idx: int,
    x: torch.Tensor,
    y: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    trainer.callbacks.call_event("on_train_step_begin", batch=batch_idx)

    trainer.optimizer.zero_grad()
    
    y_pred = trainer.model(x)
    loss = trainer.criterion(y_pred, y)

    loss.backward()
    trainer.optimizer.step()

    trainer.metrics.step(
        batch_idx=batch_idx,
        y_true=y.detach(), 
        y_pred=y_pred.detach(),
    )

    #trainer.loggers.log(train_loss=loss.detach(), 
    #        **metrics.stepped_values(batch_idx, prefix='train_'))
    trainer.callbacks.call_event("on_train_step_end", 
            batch=batch_idx, **trainer.metrics.stepped_values(batch_idx))


