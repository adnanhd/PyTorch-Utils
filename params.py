import copy, torch, argparse

class HParams(object):
    """Hyperparameters used for training."""
    def __init__(self, *args, **kwargs):
        ### wandb config
        self.project = None
        self.entity = None

        self.num_epochs = 1
        self.batch_size = 1
        self.learn_rate = 4e-5
        
        ## Trainer params
        self.save_loss = False
        self.plot_loss = False
        self.verbose = True
        
        self.dtype = torch.float
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.save_path = "./model"
        self.plot_path = None
        self.loss_path = None
        
        ### fitting parameters
        self.xtype = torch.float
        self.ytype = torch.float
        
        self.save_iter = False
    
        ### evaluating parameters
        self.eval_steps = None  # All instances in the test set are evaluated.
        self.load_iter = False

        for key, value in kwargs.items():
            if key in self.__dict__.keys():
                self.__setattr__(key, value)

        for key, value in zip(filter(lambda k: k not in kwargs.keys(), self.__dict__.keys()), args):
            if key in self.__dict__.keys():
                self.__setattr__(key, value)

    @property
    def lr(self):
        return self.learn_rate

    @property
    def epochs(self):
        return self.num_epochs
    
    @property
    def wandb(self):
        return {key: copy.deepcopy(self.__getattribute__(key)) 
                for key in ('epochs', 'batch_size', 'lr', 'device')}

    @property
    def fit(self):
        return {key: copy.deepcopy(self.__getattribute__(key)) 
                for key in ('save_iter', 'save_loss', 'plot_loss', 'verbose')}

    @property
    def evaluate(self):
        return {key: copy.deepcopy(self.__getattribute__(key)) 
                for key in ('load_iter', 'save_loss', 'plot_loss', 'verbose')}

    def __getitem__(self, index):
        return self.__getattribute__(index)

    def __len__(self):
        return len(self.__dict__)

    def __repr__(self):
        delimiter = ",\n" + " " * 7
        return "Config({})".format(delimiter.join("{}=\"{}\"".format(key, value.__repr__())
            for key, value in self.__dict__.items() if value))

    @classmethod
    def from_parser(cls, args: argparse.ArgumentParser):
        return cls(**vars(args.parse_args()))

