import os, time, torch
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
    ax.hist(metrics['Test Loss'].cpu().tolist(), bins=len(metrics['Test Loss']) // 10)
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
    def __init__(self, model, loss=None, optim=None, 
            save_path=None, plot_path=None, loss_path=None, 
            device=None, xtype=None, ytype=None):
        
        # where state dicts are stored
        self.save_path=save_path if save_path else '../Save'
        # where loss dicts and train test logs are stored
        self.loss_path=loss_path if loss_path else os.path.join(self.save_path, 'loss')
        # where dataset visualizations are saved (png, etc.)
        self.plot_path=plot_path if plot_path else os.path.join(self.save_path, 'plot')
        
        if device:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if xtype:
            self.xtype = xtype
        else:
            self.xtype = torch.float
        
        if ytype:
            self.ytype = ytype
        else:
            self.ytype = self.xtype
        
        self.model=model.to(device=device, dtype=xtype)
        self.loss=loss.to(device=device, dtype=xtype)
        self.optimizer=optim

        recursive_mkdir(self.save_path)
        recursive_mkdir(self.plot_path)
        recursive_mkdir(self.loss_path)

    def save_model(self, epoch, path=None, train_loss=None, valid_loss=None, test_loss=None):
        if not path:
            path = self.save_path
        save_path = os.path.join(path, 'checkpoints_{}_iter.ckpt'.format(epoch))
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss_func': self.loss.state_dict(),
            'train_loss': torch.as_tensor(train_loss) if train_loss else None,
            'valid_loss': torch.as_tensor(valid_loss) if valid_loss else None,
            'test_loss': torch.as_tensor(test_loss) if test_loss else None
        }
        torch.save(state, save_path)

    def save_loss(self, fname, loss, epoch, path=None, ax=None):
        if not path:
            path = self.loss_path
        save_path = os.path.join(path, '{}_loss_{}_iter.ckpt'.format(fname, epoch))
        if ax:
            if 'train' in fname or 'valid' in fname:
                ax.semilogy(range(epoch), loss, label=fname)
        
            elif 'test' in fname:
                ax.hist(loss, bins=20)
        if not isinstance(loss, torch.Tensor):
            loss = torch.as_tensor(loss)
        
        torch.save(torch.as_tensor(loss), save_path)


    def load_model(self, epoch=None, path=None):
        if not path:
            path = self.save_path
        if not epoch:
            epoch = max(int(p.split('_')[1]) for p in os.listdir(path) if 'checkpoints' in p)

        load_path = os.path.join(path, 'checkpoints_{}_iter.ckpt'.format(epoch))
        checkpoint = torch.load(load_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.loss.load_state_dict(checkpoint['loss_func'])
        return epoch, checkpoint['train_loss'], checkpoint['valid_loss'], checkpoint['test_loss']

    def load_loss(self, fname, epoch=None, path=None):
        if not path:
            path = self.loss_path
        if not epoch:
            epoch = max(int(p.split('_')[2]) for p in os.listdir(path) if 'loss' in p)
        load_path = os.path.join(path, '{}_loss_{}_iter.ckpt'.format(fname, epoch))
        return torch.load(load_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def fit(self, epochs, train_dataset, 
            valid_dataset=None, 
            save_iter=False, # save model and losses
            save_loss=False, # save model and losses
            plot_loss=False, # plot losses
            verbose=True, # print and save logs
            callbacks=[]):
        train_loss_lst = torch.empty(epochs, device=self.device)
        valid_loss_lst = torch.empty(epochs, device=self.device)

        metrics={'Train Loss': '.##e#', 'Valid Loss': '.##e#'}
        if verbose and visual:
            progress_bar = trange(
                epochs,
                unit='epoch',
                file=os.sys.stdout,
                dynamic_ncols=True,
                desc='Train',
                ascii=True,
                postfix=metrics)

        else:
            progress_bar = range(epochs)

        for epoch in progress_bar:

            loss_list = torch.zeros(len(train_dataset), device=self.device)
            for i, (features, y_true) in enumerate(train_dataset):
                y_true = y_true.to(device=self.device, dtype=self.ytype)
                features = features.to(device=self.device, dtype=self.xtype)

                y_pred = self.model(features)
                loss = self.loss(y_pred, y_true)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                loss_list[i] = loss.detach()
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

                    loss = self.loss(y_pred, y_true)

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
                self.save_model(epoch + 1)

        metrics['Train Loss'] = train_loss_lst.cpu()
        metrics['Valid Loss'] = valid_loss_lst.cpu()
        
        if plot_loss:
            subplot_train(self, metrics)
        
        if save_loss:
            self.save_loss(fname='train', loss=train_loss_lst.cpu().tolist(), epoch=epochs)
            self.save_loss(fname='valid', loss=valid_loss_lst.cpu().tolist(), epoch=epochs)

        return metrics

    def evaluate(self, test_dataset, load_iter=False,
            save_loss=False, # save model and losses
            plot_loss=False, # plot losses
            verbose=True, # print and save logs
            callbacks=[]):
        test_loss_lst = torch.zeros(len(test_dataset), device=self.device)

        if load_iter:
            if isinstance(load_iter, bool):
                load_iter=self.load_model()[0]
            else:
                load_iter=self.load_model(load_iter)[0]

        metrics = {'Test Loss': 0, 'Epochs': load_iter}
        
        if verbose and visual:
            test_dataset = tqdm(
                test_dataset,
                unit='case',
                file=os.sys.stdout,
                dynamic_ncols=True,
                desc='Evaluate',
                postfix=metrics)

        self.model.eval()
        with torch.no_grad():
            for i, (features, y_true) in enumerate(test_dataset):
                y_true = y_true.to(device=self.device, dtype=self.ytype)
                features = features.to(device=self.device, dtype=self.xtype)

                y_pred = self.model(features)
                loss = self.loss(y_pred, y_true)
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
            self.save_loss(fname='test', loss=test_loss_lst, epoch=load_iter, path=self.loss_path)

        return metrics



