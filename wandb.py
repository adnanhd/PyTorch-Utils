import torch

class HParams(object):
    """Hyperparameters used for training."""
    def __init__(self, *args, **kwargs):
        ### wandb config
        self.epochs = 1
        self.batch_size = 1
        self.lr = 4e-5
        
        self.plot_loss = False
        self.save_loss = False
        self.verbose = True
        
        ### training parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dtype = torch.float
        self.xtype = torch.float
        self.ytype = torch.float
       
        self.save_path = None
        self.plot_path = None
        
        self.save_iter = False

        for key, value in kwargs.items():
            self.__setattr__(key, value)

        for key, value in zip(filter(lambda k: k not in kwargs.keys(), self.__dict__.keys()), args):
            self.__setattr__(key, value)
    
        ### evaluating parameters
        self.eval_steps = None  # All instances in the test set are evaluated.
        self.load_iter = False
    
    
    @property
    def wandb(self):
        return {key: self.__getattribute__(key) for key in ('epochs', 'batch_size', 'lr')}

    @property
    def fit(self):
        return {key: self.__getattribute__(key) for key in ('save_iter', 'save_loss', 'plot_loss', 'verbose')}

    @property
    def evaluate(self):
        return {key: self.__getattribute__(key) for key in ('load_iter', 'save_loss', 'plot_loss', 'verbose')}

    def __getitem__(self, index):
        return self.__getattribute__(index)

    def __len__(self):
        return len(self.__dict__)

    def __repr__(self):
        return "Config({})".format(", ".join("{}=\"{}\"".format(key, value) 
            for key, value in self.__dict__.items() if value))


