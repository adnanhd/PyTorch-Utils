import pandas as pd
import os
import time
import torch
import math
from .metrics import loss_to_metric
from tqdm import tqdm, trange


def makedirs(path, verbose=False):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


class Trainer:
    def __init__(self,  model, loss=None, optim=None, sched=None, loss_path=None,
                 model_path=None, device=None, xtype=None, ytype=None, model_name=None,
                 *args, **kwargs):

        self.model_path = model_path if model_path else 'model'
        self.loss_path = loss_path if loss_path else os.path.join(
            self.model_path, 'loss')

        if device:
            self.device = device
        else:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')

        self.xtype = xtype if xtype else torch.float
        self.ytype = ytype if ytype else torch.float

        self.model = model.to(device=self.device, dtype=self.xtype)
        self.loss_func = loss
        self.optimizer = optim
        self.scheduler = sched

        self._stop_iter = True

        self.model_name = model_name if model_name is not None else 'checkpoints'

    # if loss is not then model saves the best results only
    def save_checkpoint(self, epoch=None, path=None, **kwargs):
        makedirs(self.model_path)
        if path is None or os.path.isdir(path):
            path = os.path.join(self.model_path if path is None else path,
                                f'{self.model_name}.ckpt')

        state = {
            'version': '1.0.0', **kwargs,
            'model': self.model.state_dict() if self.model else None,
            'optimizer': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'criterion': self.loss_func.state_dict() if self.loss_func else None,
        }

        torch.save(state, path)

    def save_metrics(self, label, epoch, path=None, **metrics):
        makedirs(self.loss_path)
        if path is None or os.path.isdir(path):
            path = os.path.join(self.loss_path if path is None else path,
                                f'{self.model_name}_{label}_loss_{epoch}_iter.ckpt')
        torch.save(metrics, path)

    def load_checkpoint(self, path=None):
        if path is None:
            path = self.model_path

        path = os.path.join(path, f'{self.model_name}.ckpt')

        checkpoint = torch.load(path, map_location=self.device)
        checkkeys = ('model', 'scheduler', 'optimizer', 'criterion')

        for key in checkkeys:
            key2 = key if key != 'criterion' else 'loss_func'
            if self.__getattribute__(key2) and checkpoint[key]:
                self.__getattribute__(key2).load_state_dict(checkpoint[key])

    def load_metrics(self, label, epoch=None, path=None):
        if path is None:
            path = self.loss_path

        if not os.path.isdir(path):
            path = os.path.split(path)[0]

        if not epoch:
            epoch = max(int(p.split('_')[2])
                        for p in os.listdir(path) if '_loss_' in p)

        path = os.path.join(
            path, f'{self.model_name}_{label}_loss_{epoch}_iter.ckpt')

        return torch.load(path, map_location=self.device)

    def stop_iter(self, restart=False):
        self._stop_iter = not restart

    def fit(self, epochs, train_dataset,
            valid_dataset=None,
            save_metrics=False,
            verbose=True,  # print and save logs
            callbacks=[],
            metrics={}):

        self._stop_iter = False
        metrics.setdefault('loss', loss_to_metric(self.loss_func))
        train_df = pd.DataFrame(columns=metrics.keys(), index=range(epochs))
        valid_df = pd.DataFrame(columns=metrics.keys(), index=range(epochs))

        train_dataset.dataset.features = train_dataset.dataset.features.to(
            device=self.device, dtype=self.xtype)
        train_dataset.dataset.labels = train_dataset.dataset.labels.to(
            device=self.device, dtype=self.ytype)

        for epoch in range(epochs):
            self.model.train()
            loss_list = torch.zeros(
                len(train_dataset), len(metrics), device=self.device)

            if verbose:
                progress_bar = tqdm(train_dataset, 
                                    unit='batch', file=os.sys.stdout,
                                    dynamic_ncols=True, desc=f'Epoch:{epoch}',
                                    ascii=True, colour='GREEN')
            else:
                progress_bar = train_dataset

            for batch, (features, y_true) in enumerate(progress_bar):
                y_pred = self.model(features)
                # y_pred[:,0,:,:][y_pred[:,0,:,:] == 0] = 1e-5  # 06.01.2022 
                y_pred[y_pred == 0] = 1e-5
                y_true[y_true == 0] = 1e-5

                loss = self.loss_func(y_pred, y_true, features)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                for metric_name, metric_func in enumerate(metrics.values()):
                    loss_list[batch, metric_name] = metric_func(
                        y_true=y_true.detach(), 
                        y_pred=y_pred.detach(), 
                        x_feat=features.detach(),
                    ).item()

                if verbose:
                    progress_bar.set_postfix(
                        **dict(zip(train_df.columns, loss_list[batch].cpu().tolist())))

            train_df.iloc[epoch] = loss_list.mean(dim=0).cpu()

            self.model.eval()
            with torch.no_grad():  # VALIDATION
                if valid_dataset is not None:
                    valid_df.iloc[epoch] = self.evaluate(
                        valid_dataset,
                        load_model=False,
                        save_metrics=False,
                        verbose=False,
                        callbacks=[],
                        metrics=metrics).mean(axis=0)

                if verbose:
                    df = pd.DataFrame((train_df.iloc[epoch], valid_df.iloc[epoch]),
                                      columns=metrics.keys(), index=['train', 'valid'])
                    print(df)

                for callback in callbacks:
                    callback.on_epoch_end(trainer=self,
                                          epoch=epoch + 1,
                                          y_pred=y_pred.detach(),
                                          y_true=y_true.detach(),
                                          x_true=features.detach(),
                                          **train_df.iloc[epoch].add_prefix('train_'),
                                          **valid_df.iloc[epoch].add_prefix('valid_')
                                          )

            if self.scheduler is not None:
                self.scheduler.step()

            if self._stop_iter:
                break

        for callback in callbacks:
            callback.on_training_end(self, epoch=epoch + 1,
                                     **train_df.iloc[epoch].add_prefix('train_'),
                                     **valid_df.iloc[epoch].add_prefix('valid_')
                                     )

        if save_metrics:
            self.save_metrics(label='train', epoch=epochs, **train_df)
            self.save_metrics(label='valid', epoch=epochs, **valid_df)

        return pd.concat((train_df.add_prefix('train_'), valid_df.add_prefix('valid_')), axis=1)

    def evaluate(self, test_dataset,
                 load_model=False,
                 save_metrics=False,  # save model and losses
                 verbose=True,  # print and save logs
                 callbacks=[], metrics={}):
        metrics.setdefault('loss', loss_to_metric(self.loss_func))
        test_df = pd.DataFrame(columns=metrics.keys(),
                               index=range(len(test_dataset)))

        if load_model is None or load_model:
            load_model = self.load_checkpoint(epoch=load_model)

        if verbose:
            test_dataset = tqdm(
                test_dataset,
                unit='case',
                file=os.sys.stdout,
                dynamic_ncols=True,
                desc=f'Test:{load_model}',
                colour='GREEN',
                postfix=metrics,
            )

        self.model.eval()
        with torch.no_grad():
            loss_list = torch.empty(len(test_dataset), len(
                test_df.columns), device=self.device)
            for i, (features, y_true) in enumerate(test_dataset):
                y_true = y_true.to(device=self.device, dtype=self.ytype)
                features = features.to(device=self.device, dtype=self.xtype)

                y_pred = self.model(features)

                y_pred[y_pred == 0] = 1e-5
                y_true[y_true == 0] = 1e-5

                loss = self.loss_func(y_pred, y_true)

                for name, metric in metrics.items():
                    test_df[name][i] = metric(
                        y_true=y_true.detach(), 
                        y_pred=y_pred.detach(),
                        x_feat=features.detach(),
                    ).item()

                for callback in callbacks:
                    callback.on_testing_end(trainer=self,
                                            y_pred=y_pred.detach(),
                                            y_true=y_true.detach(),
                                            x_true=features.detach(),
                                            # **test_df.iloc[i].add_prefix('test_')
                                            )

                if verbose:
                    test_dataset.set_postfix(**test_df.iloc[i])

        if save_metrics:
            self.save_metrics(label='test', epoch=load_model, **test_df)

        return test_df
