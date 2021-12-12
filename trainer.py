import pandas as pd
import os, time, torch, math
#from .plots import subplot_train, subplot_test
from tqdm import tqdm, trange

def makedirs(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

class Trainer:
    def __init__(self, model, loss=None, optim=None, sched=None, loss_path=None,
            save_path=None, device=None, xtype=None, ytype=None, *args, **kwargs):
        
        self.save_path=save_path if save_path else 'model'
        self.loss_path=loss_path if loss_path else os.path.join(self.save_path, 'loss')
        
        if device:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.xtype = xtype if xtype else torch.float
        self.ytype = ytype if ytype else torch.float
        
        self.model = model.to(device=device, dtype=xtype)
        self.loss_func = loss
        self.optimizer = optim
        self.scheduler = sched

        self._stop_iter = True

        makedirs(self.save_path)
        makedirs(self.loss_path)

    # if loss is not then model saves the best results only
    def save_checkpoint(self, epoch=None, path=None, best_metric=None):
        if path is None or os.path.isdir(path):
            path = os.path.join(self.save_path if path is None else path, 
                                'checkpoints_{}_iter.ckpt'.format(epoch))
            
        if best_metric is not None and os.path.isdir(path) and \
                best_metric < torch.load(path).get('best_metric', float('Inf')) or best_metric is None:
            state = {
               'model': self.model.state_dict() if self.model else None,
               'optimizer': self.optimizer.state_dict() if self.optimizer else None,
               'scheduler': self.scheduler.state_dict() if self.scheduler else None,
               'loss_func': self.loss_func.state_dict() if self.loss_func else None,
               'best_metric': best_metric,
            }

            torch.save(state, path)

    def save_metrics(self, label, epoch, path=None, **metrics):
        if path is None or os.path.isdir(path):
            path = os.path.join(self.loss_path if path is None else path, 
                                '{}_loss_{}_iter.ckpt'.format(label, epoch))
        torch.save(metrics, path)

    def load_checkpoint(self, epoch=None, path=None):
        if path is None:
            path = self.save_path

        if not os.path.isdir(path):
            path = os.path.split(path)[0]

        if epoch is None or isinstance(epoch, bool) and epoch:
            epoch = max(int(p.split('_')[1]) for p in os.listdir(path) if 'checkpoints' in p)

        path = os.path.join(path, f'checkpoints_{load_epoch}_iter.ckpt')
        
        checkpoint = torch.load(path, map_location=self.device)
        checkkeys = ('model', 'scheduler', 'optimizer', 'loss_func')

        for key in checkkeys:
            if self.__getattribute__(key) and checkpoint[key]:
                self.__getattribute__(key).load_state_dict(checkpoint[key])
                #del checkpoint[key]
        
        return load_epoch#, pd.DataFrame(checkpoint, columns=checkpoint.keys(), index=range(epoch))
    
    def load_metrics(self, label, epoch=None, path=None):
        if path is None:
            path = self.loss_path

        if not os.path.isdir(path):
            path = os.path.split(path)[0]

        if not epoch:
            epoch = max(int(p.split('_')[2]) for p in os.listdir(path) if 'loss' in p)
        
        path = os.path.join(path, f'{label}_loss_{epoch}_iter.ckpt')

        return torch.load(path, map_location=self.device)

    def stop_iter(self, restart=False):
        self._stop_iter = not restart

    def fit(self, epochs, train_dataset, 
            valid_dataset=None, 
            load_model=None,
            save_model=False, # save model and losses
            verbose=True, # print and save logs
            callbacks=[], 
            metrics={}):
        
        self._stop_iter = False
        metrics.setdefault('loss', self.loss_func)
        train_df = pd.DataFrame(columns=metrics.keys(), index=range(epochs))
        valid_df = pd.DataFrame(columns=metrics.keys(), index=range(epochs))

        if load_model is not None:
            load_model = self.load_checkpoint(epoch=load_model)
            train_df.update(self.load_metrics(label='train', epoch=load_model, path=None))
            valid_df.update(self.load_metrics(label='valid', epoch=load_model, path=None))
        else:
            load_model = 0

        if isinstance(save_model, int) and save_model <= 0:
            save_model = False

        for epoch in range(epochs):
            self.model.train()
            loss_list = torch.zeros(len(train_dataset), len(metrics), device=self.device)

            if verbose:
                progress_bar = tqdm(train_dataset, unit='batch', 
                        initial=load_model, file=os.sys.stdout, 
                        dynamic_ncols=True, desc=f'Epoch:{epoch}', 
                        ascii=True, colour='GREEN')
            else:
                progress_bar = range(train_dataset)
            
            for i, (features, y_true) in enumerate(progress_bar):
                y_true = y_true.to(device=self.device, dtype=self.ytype)
                features = features.to(device=self.device, dtype=self.xtype)

                y_pred = self.model(features)
                loss = self.loss_func(y_pred, y_true)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                for m, metric in enumerate(metrics.values()):
                    loss_list[i, m] = metric(y_pred, y_true).detach()
                
                if verbose:
                    progress_bar.set_postfix(**dict(zip(train_df.columns, loss_list[i].cpu().tolist())))
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            train_df.iloc[epoch] = loss_list.mean(dim=0).cpu()

            self.model.eval()
            with torch.no_grad():  # VALIDATION
                """
                loss_list = torch.zeros(len(valid_dataset), len(valid_df.columns), device=self.device)
                for i, (features, y_true) in enumerate(valid_dataset):
                    y_true = y_true.to(device=self.device, dtype=self.ytype)
                    features = features.to(device=self.device, dtype=self.xtype)

                    y_pred = self.model(features)
                    loss = self.loss_func(y_pred, y_true)
                
                    for m, metric in enumerate(metrics.values()):
                        loss_list[i, m] = metric(y_pred, y_true).detach()
                
                loss_list = loss_list.mean(dim=0).cpu()
                """
                valid_df.iloc[epoch] = self.evaluate(
                        valid_dataset, 
                        load_model=False, 
                        save_metrics=False,
                        verbose=False, 
                        callbacks=[], 
                        metrics=metrics).mean(axis=0)

                if verbose:
                    progress_bar.set_postfix(
                            **train_df.iloc[epoch].add_prefix('train_'),
                            **valid_df.iloc[epoch].add_prefix('valid_'))
                
                for callback in callbacks:
                    callback.on_epoch_end(self, epoch=epoch + 1,
                        y_pred=y_pred.detach().cpu(),
                        y_true=y_true.detach().cpu(),
                        x_true=features.detach().cpu(),
                      **train_df.iloc[epoch].add_prefix('train_'),
                      **valid_df.iloc[epoch].add_prefix('valid_')
                    )

            #if save_model and (epoch + 1) % save_iter == 0:
            #    self.save_checkpoint(epoch=epoch + 1, **valid_df if best_only else None)
            
            if self._stop_iter:
                break
        
        for callback in callbacks:
            callback.on_training_end(self, epoch=epoch + 1,
              **train_df.iloc[epoch].add_prefix('train_'),
              **valid_df.iloc[epoch].add_prefix('valid_')
            )
            
        if save_model:
            self.save_checkpoint(epoch=epochs)
            self.save_metrics(label='train', epoch=epochs, **train_df)
            self.save_metrics(label='valid', epoch=epochs, **valid_df)

        return pd.concat((train_df.add_prefix('train_'), valid_df.add_prefix('valid_')), axis=1)

    def evaluate(self, test_dataset, 
            load_model=False,
            save_metrics=False, # save model and losses
            verbose=True, # print and save logs
            callbacks=[], metrics={}):
        metrics.setdefault('loss', self.loss_func)
        test_df = pd.DataFrame(columns=metrics.keys(), index=range(len(test_dataset)))

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
            loss_list = torch.empty(len(test_dataset), len(test_df.columns), device=self.device)
            for i, (features, y_true) in enumerate(test_dataset):
                y_true = y_true.to(device=self.device, dtype=self.ytype)
                features = features.to(device=self.device, dtype=self.xtype)

                y_pred = self.model(features)
                loss = self.loss_func(y_pred, y_true)

                for name, metric in metrics.items():
                    test_df[name][i] = metric(y_pred, y_true).detach().cpu().item()
                
                for callback in callbacks:
                    callback.on_testing_end(self, i,
                        y_pred=y_pred.detach().cpu(),
                        y_true=y_true.detach().cpu(),
                        x_true=features.detach().cpu(),
                      **test_df.iloc[i].add_prefix('test_')
                    )

                if verbose:
                    test_dataset.set_postfix(**test_df.iloc[i])

        if save_metrics:
            self.save_metrics(label='test', epoch=epochs, **test_df)

        return test_df



