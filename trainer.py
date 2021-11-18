import os, time, torch, math, wandb
import matplotlib.pyplot as plt
try:
    from tqdm import tqdm, trange
except ImportError:
    visual = False
else:
    visual = True

def recursive_mkdir(path):
    try:
        os.mkdir(path)
    except FileNotFoundError:
        parent, child = os.path.split(path)
        helper_mkdir(parent)
        os.mkdir(child)
    except FileExistsError:
        pass


def subplot_train(trainer, metrics):
    fig, ax = subplots()
    epochs = len(metrics['Train Loss'])
    ax.set_title('Training and Validation Loss in {} Iter'.format(epochs))
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.semilogy(range(epochs), metrics['Train Loss'].cpu().tolist(), label='train')
    ax.semilogy(range(epochs), metrics['Valid Loss'].cpu().tolist(), label='valid')
    ax.legend()
    plt.savefig(os.path.join(trainer.plot_path, "train_loss_logy_{}_iter.png".format(epochs)))
    plt.close(fig)


def subplot_test(trainer, metrics):
    fig, ax = plt.subplots()
    ax.set_title('Testing Loss in {} Case'.format(len(metrics['Test Loss'])))
    ax.set_ylabel('Cases')
    ax.set_xlabel('Loss (avg. {:.3e})'.format(torch.as_tensor(metrics['Test Loss']).mean()))
    ax.hist(metrics['Test Loss'].cpu().tolist(), bins=math.log2(len(metrics['Test Loss'])) ** 2)
    plt.savefig(os.path.join(trainer.plot_path, 'test_loss_hist_{}_iter.png'.format(metrics['Epochs'])))
    plt.close(fig)


def subplots(ylabel=None, xlabel=None, title=None, ax=None, fig=None):
    if not ax:
        fig, ax = plt.subplots()
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)

    return fig, ax


class Trainer:
    def __init__(self, model, loss=None, optim=None, sched=None, 
            save_path=None, plot_path=None, loss_path=None, 
            device=None, xtype=None, ytype=None, wandb=None):
        
        # where state dicts are stored
        self.save_path=save_path if save_path else 'model'
        # where loss dicts and train test logs are stored
        self.loss_func_path=loss_path if loss_path else os.path.join(self.save_path, 'loss')
        # where dataset visualizations are saved (png, etc.)
        self.plot_path=plot_path if plot_path else os.path.join(self.save_path, 'plot')
        
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

        recursive_mkdir(self.save_path)
        recursive_mkdir(self.plot_path)
        recursive_mkdir(self.loss_func_path)

        if (wandb):
            self.wandb = wandb.init(project='eeg-gcn', entity='louisccc',
                config={
                    "num_epochs": num_epochs,
                    "learning_rate": learning_rate,
                    "weight_decay_ratio": weight_decay_ratio,
                    "batch_size": batch_size,
                    "device": device,
                    "num_layers": num_layers,
                    "layer_spec": layer_spec,
                    "initial_dim": cfg.initial_dim,
                    "dropout": cfg.dropout,
                    "num_classes": cfg.num_classes,
                    "wandb": wandb
                })
            self.wandb.watch(self.model, log="all")
        else:
            self.wandb = None


    def save_checkpoint(self, epoch=None, path=None, **kwargs):
        if path is None or os.path.isdir(path):
            path = os.path.join(self.save_path if path is None else path, 
                                'checkpoints_{}_iter.ckpt'.format(epoch))
            
        state = {
            'model': self.model.state_dict() if self.model else None,
            'optimizer': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'loss_func': self.loss_func.state_dict() if self.loss_func else None, **kwargs
        }
        torch.save(state, path)

    def save_loss(self, label, loss, epoch, path=None):
        if path is None or os.path.isdir(path):
            path = os.path.join(self.loss_func_path if path is None else path, 
                                '{}_loss_{}_iter.ckpt'.format(label, epoch))
        
        """
        if logy is not None and isinstance(logy, plt.Subplot):
            logy.semilogy(range(epoch), loss, label=label)

        if hist is not None and isinstance(hist, plt.Subplot):
            hist.hist(loss, bins=20)

        if plot is not None and isinstance(plot, plt.Subplot):
            plot.plot(range(epoch), loss, label=label)
        """
        
        if not isinstance(loss, torch.Tensor):
            loss = torch.as_tensor(loss)
        
        torch.save(loss, path)


    def load_checkpoint(self, epoch=None, path=None):
        if path is None:
            path = self.save_path

        if not os.path.isdir(path):
            path = os.path.split(path)[0]

        if epoch is None:
            epoch = max(int(p.split('_')[1]) for p in os.listdir(path) if 'checkpoints' in p)
        
        path = os.path.join(path, 'checkpoints_{}_iter.ckpt'.format(epoch))
        
        checkpoint = torch.load(path, map_location=self.device)
        checkkeys = ('model', 'scheduler', 'optimizer', 'loss_func')

        for key in checkkeys:
            if self.__getattribute__(key) and checkpoint[key]:
                self.__getattribute__(key).load_state_dict(checkpoint[key])

        return {'epochs': epoch, **{key: value for key, value in checkpoint.items() 
                                               if  key    not in checkkeys}}

    def load_loss(self, label, epoch=None, path=None):
        if path is None:
            path = self.loss_func_path

        if not os.path.isdir(path):
            path = os.path.split(path)[0]

        if not epoch:
            epoch = max(int(p.split('_')[2]) for p in os.listdir(path) if 'loss' in p)
        
        path = os.path.join(path, '{}_loss_{}_iter.ckpt'.format(label, epoch))

        return torch.load(path, map_location=self.device)

    def fit(self, epochs, train_dataset, 
            valid_dataset=None, 
            load_iter=None,
            save_iter=False, # save model and losses
            save_loss=False, # save model and losses
            plot_loss=False, # plot losses
            verbose=True, # print and save logs
            callbacks=[]):
        train_loss_lst = torch.empty(epochs, device=self.device)
        valid_loss_lst = torch.empty(epochs, device=self.device)

        metrics={'Train Loss': None, 'Valid Loss': None}

        if load_iter is not None:
            load_iter = self.load_checkpoint(epoch=load_iter)['epochs']
            train_loss_lst[:load_iter] = self.load_loss(label='train', epoch=load_iter)
            valid_loss_lst[:load_iter] = self.load_loss(label='valid', epoch=load_iter)
        else:
            load_iter = 0

        progress_bar = range(epochs)
            
        if verbose and visual:
            progress_bar = tqdm(
                progress_bar,
                unit='epoch',
                initial=load_iter,
                file=os.sys.stdout,
                dynamic_ncols=True,
                desc='Train',
                ascii=True,
                colour='GREEN',
                postfix=metrics)

        self.loss_func = self.loss_func.to(device=self.device, dtype=self.xtype)

        for epoch in progress_bar:

            loss_list = torch.zeros(len(train_dataset), device=self.device)
            for i, (features, y_true) in enumerate(train_dataset):
                y_true = y_true.to(device=self.device, dtype=self.ytype)
                features = features.to(device=self.device, dtype=self.xtype)

                y_pred = self.model(features)
                loss = self.loss_func(y_pred, y_true)
                loss_list[i] = loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                loss_list[i] = loss.detach()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            metrics['Train Loss'] = "{:.3e}".format(loss_list.mean())
            train_loss_lst[epoch] = loss_list.mean()

            if verbose and visual:
                progress_bar.set_postfix(**metrics)

            self.model.eval()
            with torch.no_grad():  # VALIDATION
                loss_list = torch.zeros(len(valid_dataset), device=self.device)
                for i, (features, y_true) in enumerate(valid_dataset):
                    y_true = y_true.to(device=self.device, dtype=self.ytype)
                    features = features.to(device=self.device, dtype=self.xtype)

                    y_pred = self.model(features)
                    loss = self.loss_func(y_pred, y_true)

                    loss_list[i] = loss.mean()

                metrics['Valid Loss'] = "{:.3e}".format(loss_list.mean())
                valid_loss_lst[epoch] = loss_list.mean()
                
            for callback in callbacks:
                sample = {
                    'predictions' : y_pred,
                    'features': features,
                    'labels': y_true,
                    'loss': loss
                }
                callback(self, **sample)
                
            if verbose and visual:
                progress_bar.set_postfix(**metrics)

            if save_iter and (epoch + 1) % int(save_iter) == 0:
                self.save_checkpoint(epoch + 1)

        metrics['Train Loss'] = train_loss_lst.cpu()
        metrics['Valid Loss'] = valid_loss_lst.cpu()
        
        if plot_loss:
            subplot_train(self, metrics)
        
        if save_loss:
            self.save_loss(label='train', loss=train_loss_lst.cpu().tolist(), epoch=epochs)
            self.save_loss(label='valid', loss=valid_loss_lst.cpu().tolist(), epoch=epochs)

        return metrics

    def evaluate(self, test_dataset, 
            load_iter=False,
            save_loss=False, # save model and losses
            plot_loss=False, # plot losses
            verbose=True, # print and save logs
            callbacks=[]):
        test_loss_lst = torch.zeros(len(test_dataset), device=self.device)

        if load_iter is None or load_iter:
            load_iter = self.load_checkpoint(epoch=load_iter)['epochs']

        metrics = {'Test Loss': 0, 'Epochs': load_iter}
        
        if verbose and visual:
            test_dataset = tqdm(
                test_dataset,
                unit='case',
                file=os.sys.stdout,
                dynamic_ncols=True,
                desc='Evaluate',
                colour='GREEN',
                postfix=metrics)

        self.model.eval()
        with torch.no_grad():
            for i, (features, y_true) in enumerate(test_dataset):
                y_true = y_true.to(device=self.device, dtype=self.ytype)
                features = features.to(device=self.device, dtype=self.xtype)

                y_pred = self.model(features)
                loss = self.loss_func(y_pred, y_true)
                test_loss_lst[i] = loss.mean()
                metrics['Test Loss'] = "{:.3e}".format(loss.mean())
                
                for callback in callbacks:
                    sample = {
                        'predictions' : y_pred,
                        'features': features,
                        'labels': y_true,
                        'loss': loss
                    }
                    callback(self, **sample)

                if verbose and visual:
                    test_dataset.set_postfix(**metrics)

        metrics['Test Loss'] = test_loss_lst.cpu()
        if plot_loss:
            subplot_test(self, metrics)

        if save_loss:
            self.save_loss(label='test', loss=metrics['Test Loss'], epoch=load_iter)

        return metrics



