import pandas as pd
import os, time
import torch, math
from .metrics import loss_to_metric, MetricHandler
from tqdm import tqdm, trange, utils
from .callbacks import (
    CallbackHandler,
    CallbackMethodNotImplementedError,
    TrainerCallback,
    
    TrainingError,
)

from typing import List, Dict, Any, Mapping, Optional, Union, Callable, Tuple, Iterable


def makedirs(path, verbose=False):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,  
        loss: Union[Callable, torch.nn.modules.loss._Loss],
        sched: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        loss_path: Optional[str] = None,  # todo: remove
        model_name: str = "model",
        model_path: Optional[str] = "./checkpoints/",
        device=None,  # todo: remove
        xtype: Union[torch.dtype, type] = torch.float32,
        ytype: Union[torch.dtype, type] = torch.float32,
        *args,
        **kwargs,
    ):
        
        self.model_path = model_path
        self.model_name = model_name

        self.loss_path = (
            loss_path if loss_path else os.path.join(self.model_path, "loss")
        )
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.xtype = xtype
        self.ytype = ytype

        self.model = model.to(device=self.device, dtype=self.xtype)
        self.criterion = loss
        self.optimizer = optim
        self.scheduler = sched

    # if loss is not then model saves the best results only
    def save_checkpoint(self, epoch=None, path=None, best_metric=None):
        makedirs(self.model_path)
        if path is None or os.path.isdir(path):
            path = os.path.join(
                self.model_path if path is None else path, f"{self.model_name}.ckpt"
            )

        if (
            best_metric is None
            or os.path.isdir(path)
            and best_metric < torch.load(path).get("best_metric", float("Inf"))
        ):
            state = {
                "model": self.model.state_dict() if self.model else None,
                "optimizer": self.optimizer.state_dict() if self.optimizer else None,
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "criterion": self.criterion.state_dict() if self.criterion else None,
                "best_metric": best_metric,
            }

            torch.save(state, path)

    def save_metrics(self, label, epoch, path=None, **metrics):
        if path is None or os.path.isdir(path):
            path = os.path.join(
                self.loss_path if path is None else path,
                f"{self.model_name}_{label}_loss_{epoch}_iter.ckpt",
            )
        torch.save(metrics, path)

    def load_checkpoint(self, epoch=None, path=None):
        if path is None:
            path = self.model_path

        if not os.path.isdir(path):
            path = os.path.split(path)[0]

        if epoch is None or isinstance(epoch, bool) and epoch:
            epoch = max(
                int(p.split("_")[1]) for p in os.listdir(path) if self.model_name in p
            )

        path = os.path.join(path, f"{self.model_name}.ckpt")

        checkpoint = torch.load(path, map_location=self.device)
        checkkeys = ("model", "scheduler", "optimizer", "criterion")

        for key in checkkeys:
            if self.__getattribute__(key) and checkpoint[key]:
                self.__getattribute__(key).load_state_dict(checkpoint[key])
                # del checkpoint[key]

        return epoch  # , pd.DataFrame(checkpoint, columns=checkpoint.keys(), index=range(epoch))

    def load_metrics(self, label, epoch=None, path=None):
        if path is None:
            path = self.loss_path

        if not os.path.isdir(path):
            path = os.path.split(path)[0]

        if not epoch:
            epoch = max(int(p.split("_")[2]) for p in os.listdir(path) if "_loss_" in p)

        path = os.path.join(path, f"{self.model_name}_{label}_loss_{epoch}_iter.ckpt")

        return torch.load(path, map_location=self.device)

    @staticmethod
    def create_dataloader(
        dataset: torch.utils.data.Dataset,
        train_mode: bool = True,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> torch.utils.data.DataLoader:
        assert isinstance(dataset, torch.utils.data.Dataset), type(dataset)
        default_dataloader_args = dict(
            {
                "dataset": dataset,
                "shuffle": train_mode,
                "pin_memory": True if torch.cuda.is_available() else False,
                "batch_size": batch_size if train_mode else dataset.__len__(),
                "num_workers": max(
                    os.cpu_count() // torch.cuda.device_count()
                    if torch.cuda.is_available()
                    else os.cpu_count(),
                    1,
                ),
            }
        )

        default_dataloader_args.update(kwargs)

        return torch.utils.data.DataLoader(**default_dataloader_args)

    def train(
        self,
        num_epochs: int,
        train_dataset: torch.utils.data.Dataset,
        valid_dataset: Optional[torch.utils.data.Dataset] = None,
        callbacks: List[TrainerCallback] = [],
        metrics: Mapping[str, Callable] = {},
        verbose: bool = True,  # print and save logs
        batch_size: int = 8,
        train_dataloader_kwargs: Optional[Mapping] = {},
        valid_dataloader_kwargs: Optional[Mapping] = {},
    ):
        metrics.setdefault("loss", loss_to_metric(self.criterion))

        callbacks = CallbackHandler(callbacks)
        train_metrics = MetricHandler("train", metrics=metrics, 
                                      index=range(num_epochs))

        self._train_dataloader = self.create_dataloader(
            dataset=train_dataset,
            train_mode=True,
            batch_size=batch_size,
            **train_dataloader_kwargs,
        )
        
        if valid_dataset is not None:
            valid_metrics = MetricHandler("valid", metrics=metrics, 
                                          index=range(num_epochs))
            self._valid_dataloader = self.create_dataloader(
                dataset=valid_dataset,
                train_mode=False,
                batch_size=batch_size,
                **valid_dataloader_kwargs,
            )
        else:
            valid_metrics = None
            self._valid_dataloader = None

        
        self._run_training(
                num_epochs=num_epochs,
                train_dataloader=self._train_dataloader,
                valid_dataloader=self._valid_dataloader,
                callbacks=callbacks,
                train_metrics=train_metrics,
                valid_metrics=valid_metrics,
                verbose=verbose,  # print and save logs
        )
        
        if valid_dataset is not None:
            return train_metrics, valid_metrics
        else:
            return train_metrics
        """ pd.concat(
            (train_df.add_prefix("train_"), valid_df.add_prefix("valid_")), axis=1
        ) """

    def evaluate(
        self,
        test_dataset,
        load_model=False,
        save_metrics=False,  # save model and losses
        verbose=True,  # print and save logs
        callbacks=[],
        metrics={},
        test_dataloader_kwargs={},
        **kwargs,
    ):
        metrics.setdefault("loss", loss_to_metric(self.criterion))
        metrics = MetricHandler(metrics)

        eval_dataloader = self.create_dataloader(
            dataset=test_dataset,
            train_mode=False,
            batch_size=test_dataset.__len__(),
            **test_dataloader_kwargs,
        )

        callbacks = CallbackHandler(callbacks)
        metrics.setdefault("loss", loss_to_metric(self.criterion))
        eval_metrics = MetricHandler(
            "test", metrics=metrics, index=range(eval_dataloader.__len__())
        )

        self.model.eval()

        self._run_evaluate(
            eval_loader=eval_dataloader,
            callbacks=callbacks,
            metrics=eval_metrics,
        )

        return eval_metrics
    
    def _run_training(
        self,
        num_epochs: int,
        callbacks: CallbackHandler,
        train_metrics: MetricHandler,
        train_dataloader: torch.utils.data.DataLoader,
        valid_metrics: Optional[MetricHandler] = None,
        valid_dataloader: Optional[torch.utils.data.DataLoader] = None,
        verbose: bool = True,  # print and save logs
    ):
        callbacks.call_event(
            "on_training_run_begin",
            self,
        )

        for epoch in range(num_epochs):

            self._run_training_epoch(
                epoch=epoch,
                train_loader=train_dataloader,
                callbacks=callbacks,
                metrics=train_metrics,
                verbose=verbose,
            )

            if valid_dataloader is not None:
                self._run_valid_epoch(
                    valid_dataloader,
                    valid_metrics,
                    callbacks,
                    verbose=verbose,
                )

        callbacks.call_event(
            "on_training_run_end",
            self,
            epoch=epoch + 1,
        )

    def _run_training_epoch(
        self,
        epoch: int,
        train_loader: torch.utils.data.DataLoader,
        callbacks: CallbackHandler,
        metrics: MetricHandler,
        verbose: bool = True,
    ) -> torch.Tensor:
        metrics.reset(trainer=self)
        callbacks.call_event(
            "on_train_epoch_begin",
            self,
            epoch=epoch + 1,
        )

        self.model.train()
        for batch, (features, y_truth) in enumerate(train_loader):
            self._run_training_step(
                epoch=epoch, batch_idx=batch,
                x=features.to(device=self.device, dtype=self.xtype),
                y=y_truth.to(device=self.device, dtype=self.ytype),
                callbacks=callbacks,
                metrics=metrics,
            )

        metrics.update(epoch=epoch)
        callbacks.call_event(
            "on_train_epoch_end",
            self,
            epoch=epoch + 1,
        )


    def _run_training_step(
        self,
        epoch: int,
        batch_idx: int,
        x: torch.Tensor,
        y: torch.Tensor,
        callbacks: CallbackHandler,
        metrics: MetricHandler,
        **kwargs,
    ) -> torch.Tensor:
        callbacks.call_event(
            "on_train_step_begin",
            self, step=batch_idx,
        )

        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        metric_results = metrics.step(
            batch_idx=batch_idx,
            y_true=y.detach(), 
            y_pred=y_pred.detach(),
        )

        callbacks.call_event(
            "on_train_step_end",
            self, step=batch_idx,
            **metric_results,
        )

    def _run_evaluate(
        self,
        index: int,
        eval_loader: torch.utils.data.DataLoader,
        callbacks: CallbackHandler,
        metrics: MetricHandler,
    ) -> torch.Tensor:
        metrics.reset(trainer=self)

        callbacks.call_event(
            "on_valid_epoch_begin",
            self,
        )

        self.model.eval()
        with torch.no_grad():
            for batch, (features, y_true) in enumerate(eval_loader):
                self._run_evaluating_step(
                    batch_idx=batch,
                    x=features.to(device=self.device, dtype=self.xtype),
                    y=y_true.to(device=self.device, dtype=self.ytype),
                    callbacks=callbacks,
                    metrics=metrics,
                )

        metrics.update(epoch=index)
        callbacks.call_event(
            "on_valid_epoch_end",
            self,
            epoch=index + 1,
        )

    def _run_evaluating_step(
        self,
        batch_idx: int,
        x: torch.Tensor,
        y: torch.Tensor,
        callbacks: CallbackHandler,
        metrics: Mapping[str, Callable],
        loss_list: List,
        **kwargs,
    ):
        callbacks.call_event(
            "on_eval_step_begin",
            self,
            step=batch_idx,
        )

        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)

        for metric_name, metric_func in enumerate(metrics.values()):
            loss_list[batch_idx, metric_name] = metric_func(
                y_truth=y.detach().cpu(), y_pred=y_pred.detach().cpu()
            ).item()

        callbacks.call_event(
            "on_eval_step_end",
            self,
            step=batch_idx,
        )
