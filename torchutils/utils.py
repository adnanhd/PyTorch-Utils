# if loss is not then model saves the best results only
def save_checkpoint(self, epoch=None, path=None, best_metric=None):
    makedirs(self.model_path)
    if path is None or os.path.isdir(path):
        path = os.path.join(self.model_path if path is None else path, 
                            f'{self.model_name}.ckpt')
        
    if best_metric is None or os.path.isdir(path) and \
            best_metric < torch.load(path).get('best_metric', float('Inf')):
        state = {
            'model': self.model.state_dict() if self.model else None,
            'optimizer': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'loss_func': self.loss_func.state_dict() if self.loss_func else None,
            'best_metric': best_metric,
        }

        torch.save(state, path)

def save_metrics(self, label, epoch, path=None, **metrics):
    makedirs(self.loss_path)
    if path is None or os.path.isdir(path):
        path = os.path.join(self.loss_path if path is None else path, 
                            f'{self.model_name}_{label}_loss_{epoch}_iter.ckpt')
    torch.save(metrics, path)

def load_checkpoint(self, epoch=None, path=None):
    if path is None:
        path = self.model_path

    if not os.path.isdir(path):
        path = os.path.split(path)[0]

    if epoch is None or isinstance(epoch, bool) and epoch:
        epoch = max(int(p.split('_')[1]) for p in os.listdir(path) if self.model_name in p)

    path = os.path.join(path, f'{self.model_name}.ckpt')
    
    checkpoint = torch.load(path, map_location=self.device)
    checkkeys = ('model', 'scheduler', 'optimizer', 'loss_func')

    for key in checkkeys:
        if self.__getattribute__(key) and checkpoint[key]:
            self.__getattribute__(key).load_state_dict(checkpoint[key])
            #del checkpoint[key]
    
    return epoch#, pd.DataFrame(checkpoint, columns=checkpoint.keys(), index=range(epoch))

def load_metrics(self, label, epoch=None, path=None):
    if path is None:
        path = self.loss_path

    if not os.path.isdir(path):
        path = os.path.split(path)[0]

    if not epoch:
        epoch = max(int(p.split('_')[2]) for p in os.listdir(path) if '_loss_' in p)
    
    path = os.path.join(path, f'{self.model_name}_{label}_loss_{epoch}_iter.ckpt')

    return torch.load(path, map_location=self.device)