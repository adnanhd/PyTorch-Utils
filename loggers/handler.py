from utils.callbacks import TrainerCallback


class ProgressBar(TrainerCallback):
    """
    A callback which visualises the state of each training and evaluation epoch using a progress bar
    """

    def __init__(self):
        self.pbar = None

    def on_train_epoch_start(self, trainer, epoch, **kwargs):
        self.pbar = tqdm(
            total=trainer._train_dataloader.__len__(),
            #disable=not trainer._accelerator.is_local_main_process,
            unit="batch",
            initial=0,
            file=os.sys.stdout,
            dynamic_ncols=True,
            desc=f"Epoch:{epoch}",
            ascii=True,
            colour="GREEN",
        )

    def on_eval_epoch_start(self, trainer, epoch, **kwargs):
        self.pbar = tqdm(
                trainer._eval_dataloader.__len__(),
                unit="case",
                file=os.sys.stdout,
                dynamic_ncols=True,
                desc=f"Test:{epoch}",
                colour="GREEN",
                postfix=metrics,
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
        self.pbar.set_postfix(
            **dict(zip(train_df.columns, loss_list[batch].cpu().tolist()))
        )

    def on_eval_epoch_end(self, trainer, **kwargs):
        self.pbar.close()
        time.sleep(0.01)
        if verbose:
            df = pd.DataFrame(
                (train_df.iloc[epoch], valid_df.iloc[epoch]),
                columns=metrics.keys(),
                index=["train", "valid"],
            )
            print(df)
