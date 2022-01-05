
class LogMetricsCallback(TrainerCallback):
    """
    A callback that logs the latest values of any metric which has been updated in
    the trainer's run history. By default, this just prints to the command line once per machine.

    Metrics prefixed with 'train' are logged at the end of a training epoch, all other
    metrics are logged after evaluation.

    This can be subclassed to create loggers for different platforms by overriding the :meth:`~LogMetricsCallback.log_metrics` method.
    """

    def on_train_epoch_end(self, trainer, **kwargs):
        metric_names = [
            metric
            for metric in trainer.run_history.get_metric_names()
            if "train" in metric
        ]

        self._log_latest_metrics(trainer, metric_names)

    def on_eval_epoch_end(self, trainer, **kwargs):
        metric_names = [
            metric
            for metric in trainer.run_history.get_metric_names()
            if "train" not in metric
        ]
        self._log_latest_metrics(trainer, metric_names)

    def _log_latest_metrics(self, trainer, metric_names):
        latest_metrics = self._get_latest_metrics(trainer, metric_names)
        self.log_metrics(trainer, latest_metrics)

    def _get_latest_metrics(self, trainer, metric_names):
        return {
            metric_name: trainer.run_history.get_latest_metric(metric_name)
            for metric_name in metric_names
        }

    def log_metrics(self, trainer, metrics: dict):
        for metric_name, metric_value in metrics.items():
            trainer.print(f"\n{metric_name}: {metric_value}")


class ProgressBarCallback(TrainerCallback):
    """
    A callback which visualises the state of each training and evaluation epoch using a progress bar
    """

    def __init__(self):
        self.pbar = None

    def on_train_epoch_start(self, trainer, **kwargs):
        self.pbar = tqdm(
            total=len(trainer._train_dataloader),
            disable=not trainer._accelerator.is_local_main_process,
        )

    def on_train_step_end(self, trainer, **kwargs):
        self.pbar.update(1)

    def on_train_epoch_end(self, trainer, **kwargs):
        self.pbar.close()
        time.sleep(0.01)

    def on_eval_epoch_start(self, trainer, **kwargs):
        self.pbar = tqdm(
            total=len(trainer._eval_dataloader),
            disable=not trainer._accelerator.is_local_main_process,
        )

    def on_eval_step_end(self, trainer, **kwargs):
        self.pbar.update(1)

    def on_eval_epoch_end(self, trainer, **kwargs):
        self.pbar.close()
        time.sleep(0.01)


class PrintProgressCallback(TrainerCallback):
    """
    A callback which prints a message at the start and end of a run,
    as well as at the start of each epoch.
    """

    def on_training_run_start(self, trainer, **kwargs):
        trainer.print("\nStarting training run")

    def on_train_epoch_start(self, trainer, **kwargs):
        trainer.print(f"\nStarting epoch {trainer.run_history.current_epoch}")
        time.sleep(0.01)

    def on_training_run_end(self, trainer, **kwargs):
        trainer.print("Finishing training run")

    def on_evaluation_run_start(self, trainer, **kwargs):
        trainer.print("\nStarting evaluation run")

    def on_evaluation_run_end(self, trainer, **kwargs):
        trainer.print("Finishing evaluation run")


class SaveBestModelCallback(TrainerCallback):
    """
    A callback which saves the best model during a training run, according to a given metric.
    The best model weights are loaded at the end of the training run.
    """

    def __init__(
        self,
        save_path="best_model.pt",
        watch_metric="eval_loss_epoch",
        greater_is_better: bool = False,
        reset_on_train: bool = True,
    ):
        """

        :param save_path: The path to save the checkpoint to. This should end in ``.pt``.
        :param watch_metric: the metric used to compare model performance. This should be accessible from the trainer's run history.
        :param greater_is_better: whether an increase in the ``watch_metric`` should be interpreted as the model performing better.
        :param reset_on_train: whether to reset the best metric on subsequent training runs. If ``True``, only the metrics observed during the current training run will be compared.
        """
        self.watch_metric = watch_metric
        self.greater_is_better = greater_is_better
        self.operator = np.greater if self.greater_is_better else np.less
        self.best_metric = None
        self.save_path = save_path
        self.reset_on_train = reset_on_train

    def on_training_run_start(self, args, **kwargs):
        if self.reset_on_train:
            self.best_metric = None

    def on_training_run_epoch_end(self, trainer, **kwargs):
        current_metric = trainer.run_history.get_latest_metric(
            self.watch_metric)
        if self.best_metric is None:
            self.best_metric = current_metric
            trainer.save_checkpoint(
                save_path=self.save_path,
                checkpoint_kwargs={self.watch_metric: self.best_metric},
            )
        else:
            is_improvement = self.operator(current_metric, self.best_metric)

            if is_improvement:
                self.best_metric = current_metric
                trainer.save_checkpoint(
                    save_path=self.save_path,
                    checkpoint_kwargs={"loss": self.best_metric},
                )

    def on_training_run_end(self, trainer, **kwargs):
        trainer.print(
            f"Loading checkpoint with {self.watch_metric}: {self.best_metric}"
        )
        trainer.load_checkpoint(self.save_path)


class EarlyStoppingCallback(TrainerCallback):
    """
    A callback which stops training early if progress is not being observed.
    """

    def __init__(
        self,
        early_stopping_patience: int = 1,
        early_stopping_threshold: float = 0.01,
        watch_metric="eval_loss_epoch",
        greater_is_better: bool = False,
        reset_on_train: bool = True,
    ):
        """

        :param early_stopping_patience: the number of epochs with no improvement after which training will be stopped.
        :param early_stopping_threshold: the minimum change in the ``watch_metric`` to qualify as an improvement, i.e. an absolute change of less than this threshold, will count as no improvement.
        :param watch_metric: the metric used to compare model performance. This should be accessible from the trainer's run history.
        :param greater_is_better: whether an increase in the ``watch_metric`` should be interpreted as the model performing better.
        :param reset_on_train: whether to reset the best metric on subsequent training runs. If ``True``, only the metrics observed during the current training run will be compared.
        """
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.watch_metric = watch_metric
        self.greater_is_better = greater_is_better
        self.early_stopping_patience_counter = 0
        self.best_metric = None
        self.operator = np.greater if self.greater_is_better else np.less
        self.reset_on_train = reset_on_train

    def on_training_run_start(self, args, **kwargs):
        if self.reset_on_train:
            self.best_metric = None
            self.early_stopping_patience_counter = 0

    def on_training_run_epoch_end(self, trainer, **kwargs):
        current_metric = trainer.run_history.get_latest_metric(
            self.watch_metric)
        if self.best_metric is None:
            self.best_metric = current_metric
        else:
            is_improvement = self.operator(current_metric, self.best_metric)
            improvement = abs(current_metric - self.best_metric)
            improvement_above_threshold = improvement > self.early_stopping_threshold

            if is_improvement and improvement_above_threshold:
                trainer.print(
                    f"\nImprovement of {improvement} observed, resetting counter. "
                )
                self.best_metric = current_metric
                self.early_stopping_patience_counter = 0
                self.__print_counter_status(trainer)
            else:
                trainer.print(
                    "No improvement above threshold observed, incrementing counter. "
                )
                self.early_stopping_patience_counter += 1
                self.__print_counter_status(trainer)

        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            raise StopTrainingError(
                f"Stopping training due to no improvement after {self.early_stopping_patience} epochs"
            )

    def __print_counter_status(self, trainer):
        trainer.print(
            f"Early stopping counter: {self.early_stopping_patience_counter}/{self.early_stopping_patience}"
        )


class TerminateOnNaNCallback(TrainerCallback):
    """
    A callback that terminates the training run if a ``NaN`` loss is observed during either training or
    evaluation.
    """

    def __init__(self):
        self.triggered = False

    def check_for_nan_after_batch(self, batch_output, step=None):
        """Test if loss is NaN and interrupts training."""
        loss = batch_output["loss"]
        if torch.isinf(loss) or torch.isnan(loss):
            self.triggered = True
            raise StopTrainingError(
                f"Stopping training due to NaN loss in {step} step")

    def on_train_step_end(self, trainer, batch_output, **kwargs):
        self.check_for_nan_after_batch(batch_output, step="training")

    def on_eval_step_end(self, trainer, batch_output, **kwargs):
        self.check_for_nan_after_batch(batch_output, step="validation")

    def on_training_run_end(self, trainer, **kwargs):
        if self.triggered:
            sys.exit("Exiting due to NaN loss")


class MoveModulesToDeviceCallback(TrainerCallback):
    """
    A callback which moves any :class:`~pytorch_accelerated.trainer.Trainer` attributes which are instances of
    :class:`torch.nn.Module` on to the appropriate device at the start of a training or evaluation run.

    .. Note::
        This does **not** include the model, as this will be prepared separately by the
        :class:`~pytorch_accelerated.trainer.Trainer`'s internal instance of :class:`accelerate.Accelerator`.

    """

    def _get_modules(self, trainer):
        return inspect.getmembers(trainer, lambda x: isinstance(x, nn.Module))

    def _move_modules_to_device(self, trainer):
        modules = self._get_modules(trainer)

        for module_name, module in modules:
            if module_name != "model":
                module.to(trainer.device)

    def on_training_run_start(self, trainer, **kwargs):
        self._move_modules_to_device(trainer)

    def on_evaluation_run_start(self, trainer, **kwargs):
        self._move_modules_to_device(trainer)


class Callback:
    def __init__(self):
        pass

    def on_training_end(self, trainer, **kwargs):
        pass

    def on_testing_end(self, trainer, **kwargs):
        pass

    def on_epoch_end(self, trainer, **kwargs):
        pass
