import torch
import numpy as np
import pandas as pd
from .trainer import Trainer
from .callbacks import (
    CallbackHandler,
    CallbackMethodNotImplementedError,
    TrainerCallback,
    StopTrainingError,
)

from typing import List, Dict, Any, Mapping, Optional, Union, Callable, Tuple, Iterable


def loss_to_metric(loss):
    def metric(y_true, y_pred):
        return loss(input=y_pred, target=y_true)

    return metric


def one_hot_decode(score):
    def decoded_score(y_pred, y_true):
        return score(y_pred=y_pred.argmax(axis=1), y_true=y_true.argmax(axis=1))

    return decoded_score


class MetricHandler(TrainerCallback):
    def __init__(
        self,
        path: str,
        metrics: Mapping[str, Callable[[torch.Tensor, torch.Tensor], float]],
        verbose: bool = False,
        index: Optional[Iterable] = None,
    ):
        self.path = path
        self.verbose = verbose
        self.metrics = metrics
        self.loss_list = None
        self._dataframe = pd.DataFrame(columns=metrics.keys(), index=index)

    # on_train_epoch_begin
    def reset(self, trainer: Trainer):
        self.loss_list = torch.empty(
            trainer._train_dataloader.__len__(),
            self.metrics.__len__(),
            dtype=trainer.ytype,
            device=trainer.device,
        )

    # on_train_epoch_end
    def update(self, epoch: int):
        self._dataframe.loc[epoch] = self.loss_list.mean(axis=0).numpy()

    # on_train_step_end
    def step(
        self,
        batch_idx: int,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:
        self.loss_list[batch_idx, :] = torch.tensor(
            [
                self.metrics[metric_name](y_true, y_pred)
                for metric_name in self.metrics.keys()
            ]
        )
        return dict(zip(self.metrics.keys(), self.loss_list[batch_idx, :]))

    # on_train_epoch_end
    def get(self, metric: str, value: float):
        idx = self._dataframe.index[self.index]
        self._dataframe[metric].iloc[idx] = value

    def save(self):
        with open(self.path, "w") as f:
            f.write("\n".join(self.logs))


def precision_score(
    y_true: torch.Tensor, y_pred: torch.Tensor, is_training=False
) -> torch.Tensor:
    """Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6

    """
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    return tp / (tp + fp + epsilon)


def recall_score(
    y_true: torch.Tensor, y_pred: torch.Tensor, is_training=False
) -> torch.Tensor:
    """Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6

    """
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    return tp / (tp + fn + epsilon)


def f1_score(
    y_true: torch.Tensor, y_pred: torch.Tensor, is_training=False
) -> torch.Tensor:
    """Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6

    """
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1


def accuracy_score(
    y_true: torch.Tensor, y_pred: torch.Tensor, is_training=False
) -> torch.Tensor:
    """Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6

    """
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    return (tp + tn) / (tp + fp + tn + fp)
