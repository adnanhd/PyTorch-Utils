import os, time, torch, math
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



def subplot_train(trainer, train_loss, valid_loss):
    fig, ax = subplots()
    epochs = len(train_loss)
    ax.set_title('Training and Validation Loss in {} Iter'.format(epochs))
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    # TODO: add 'x' and 'o' for train and valid
    ax.semilogy(range(epochs), train_loss, label='train')
    ax.semilogy(range(epochs), valid_loss, label='valid')
    ax.legend()
    plt.savefig(os.path.join(trainer.plot_path, "train_loss_logy_{}_iter.png".format(epochs)))
    plt.close(fig)


def subplot_test(trainer, epochs, test_loss):
    fig, ax = plt.subplots()
    ax.set_title('Testing Loss in {} Case'.format(len(test_loss))
    ax.set_ylabel('Cases')
    ax.set_xlabel('Loss (avg. {:.3e})'.format(torch.as_tensor(test_loss).mean()))
    ax.hist(test_loss, bins=int(math.log2(len(test_loss) ** 2))
    plt.savefig(os.path.join(trainer.plot_path, 'test_loss_hist_{}_iter.png'.format(epochs))
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
            device=None, xtype=None, ytype=None, *args, **kwargs):
        
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

        self._stop_iter = True

        recursive_mkdir(self.save_path)
        recursive_mkdir(self.plot_path)
        recursive_mkdir(self.loss_func_path)

    # if loss is not then model saves the best results only
    def save_checkpoint(self, epoch=None, path=None, loss=None, **kwargs):
        if path is None or os.path.isdir(path):
            path = os.path.join(self.save_path if path is None else path, 
                                'checkpoints_{}_iter.ckpt'.format(epoch))
            
        if loss is not None and os.path.isdir(path) and \
                loss < torch.load(path).get('valid_loss', float('Inf')) or loss is None:
            state = {
                'model': self.model.state_dict() if self.model else None,
                'optimizer': self.optimizer.state_dict() if self.optimizer else None,
                'scheduler': self.scheduler.state_dict() if self.scheduler else None,
                'loss_func': self.loss_func.state_dict() if self.loss_func else None, 
                'valid_loss' : loss.mean().detach().cpu() if loss else None, **kwargs
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

        if epoch is None or isinstance(epoch, bool) and epoch:
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

    def stop_iter(self, restart=False):
        self._stop_iter = not restart

    def fit(self, epochs, train_dataset, 
            valid_dataset=None, 
            load_iter=None,
            save_iter=False, # save model and losses
            save_loss=False, # save model and losses
            plot_loss=False, # plot losses
            best_only=True, # saves only best models
            verbose=True, # print and save logs
            callbacks=[]):
        self._stop_iter = False
        train_loss_lst = torch.empty(epochs, device=self.device)
        valid_loss_lst = torch.empty(epochs, device=self.device)

        if load_iter is not None:
            load_iter = self.load_checkpoint(epoch=load_iter)['epochs']
            train_loss_lst[:load_iter] = self.load_loss(label='train', epoch=load_iter)
            valid_loss_lst[:load_iter] = self.load_loss(label='valid', epoch=load_iter)
        else:
            load_iter = 0

        if isinstance(save_iter, int) and save_iter <= 0:
            save_iter = False

        progress_bar = range(epochs)

        if verbose:
            try:
                progress_bar = tqdm(
                    progress_bar,
                    unit='epoch',
                    initial=load_iter,
                    file=os.sys.stdout,
                    dynamic_ncols=True,
                    desc='Train',
                    ascii=True,
                    #colour='GREEN',
                    )
            except:
                    visual = False
            else:
                    visual = True
        #self.loss_func = self.loss_func.to(device=self.device, dtype=self.xtype)

        for epoch in progress_bar:

            loss_list = torch.zeros(len(train_dataset), device=self.device)
            for i, (features, y_true) in enumerate(train_dataset):
                y_true = y_true.to(device=self.device, dtype=self.ytype)
                features = features.to(device=self.device, dtype=self.xtype)

                y_pred = self.model(features)
                loss = self.loss_func(y_pred, y_true)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                loss_list[i] = loss.detach()
            
            if self.scheduler is not None:
                self.scheduler.step(epoch)
            
            
            train_loss_lst[epoch] = loss_list.mean()

            self.model.eval()
            with torch.no_grad():  # VALIDATION
                loss_list = torch.zeros(len(valid_dataset), device=self.device)
                for i, (features, y_true) in enumerate(valid_dataset):
                    y_true = y_true.to(device=self.device, dtype=self.ytype)
                    features = features.to(device=self.device, dtype=self.xtype)

                    y_pred = self.model(features)
                    loss = self.loss_func(y_pred, y_true)

                    loss_list[i] = loss.mean()

                valid_loss_lst[epoch] = loss_list.mean()

                if visual:
                    progress_bar.set_postfix(
                        train_loss=train_loss_lst[epoch].cpu().item(),
                        valid_loss=valid_loss_lst[epoch].cpu().item())
                elif verbose:
                    print("train_loss:", train_loss_lst[epoch].cpu().item(),
                          "valid_loss:", valid_loss_lst[epoch].cpu().item())
                
            for callback in callbacks:
                callback(self, epoch=epoch + 1,
                    train_loss=train_loss_lst[epoch].clone(),
                    valid_loss=valid_loss_lst[epoch].clone(),
                    y_pred=y_pred.detach().clone(),
                    y_true=y_true.detach().clone(),
                    x_true=features.detach().clone()
                )

            if isinstance(save_iter, int) and (epoch + 1) % save_iter == 0:
                self.save_checkpoint(epoch=epoch + 1, loss=loss_list.mean() if best_only else None)
            
            if self._stop_iter:
                break

        if plot_loss:
            subplot_train(self, train_loss=train_loss_lst.cpu().tolist(), valid_loss=valid_loss_lst.cpu().tolist())
            
        if isinstance(save_iter, bool) and save_iter:
            self.save_checkpoint(epoch=load_iter + 1, loss=valid_loss_lst[-1] if best_only else None)
        
        if save_loss:
            self.save_loss(label='train', loss=train_loss_lst.cpu().tolist(), epoch=epochs)
            self.save_loss(label='valid', loss=valid_loss_lst.cpu().tolist(), epoch=epochs)

        return None

    def evaluate(self, test_dataset, 
            load_iter=False,
            save_loss=False, # save model and losses
            plot_loss=False, # plot losses
            verbose=True, # print and save logs
            callbacks=[]):
        test_loss_lst = torch.zeros(len(test_dataset), device=self.device)

        if load_iter is None or load_iter:
            load_iter = self.load_checkpoint(epoch=load_iter)['epochs']

        if verbose and visual:
            test_dataset = tqdm(
                test_dataset,
                unit='case',
                file=os.sys.stdout,
                dynamic_ncols=True,
                desc='Evaluate',
                #colour='GREEN',
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
                        'predictions' : y_pred.detach().cpu(),
                        'features': features.detach().cpu(),
                        'labels': y_true.detach().cpu(),
                        'loss': loss.detach().cpu()
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



