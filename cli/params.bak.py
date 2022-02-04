import copy, torch
from argparse import ArgumentParser


class HParams:

    """Hyperparameters used for training."""
    def __init__(self, *args, **kwargs):

        ### wandb config
        self.project = None
        self.entity = None
        
        ## Trainer params
        self.save_model = False
        self.load_model = False
        self.best_model = False
        self.model_path = "checkpoints"
        self.model_name = 'network'

        self.dtype = torch.float
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.xtype = torch.float
        self.ytype = torch.float
        
        self.verbose = True
        
        ### training parameters
        self.num_epochs = 100
        self.batch_size = 16
        self.learn_rate = 1e-4
        self.lr_decay = 0.96
        self.weight_decay = 0.0000
    
        ### evaluating parameters
        self.load_model = None  # All instances in the test set are evaluated.
        self.save_metrics = False # save metrics
        
        self.update(*args, **kwargs)


    def update(self, *args, **kwargs):
        for key, value in kwargs.items():
            #if key in self.__dict__.keys():
            self.__setattr__(key, value)

        for key, value in zip(filter(lambda k: k not in kwargs.keys(), self.__dict__.keys()), args):
            #if key in self.__dict__.keys():
            self.__setattr__(key, value)

    @property
    def dict(self):
        return self.__dict__

    @property
    def experiment(self):
        return f'{self.project}_{self.model_name}_{self.entity}'

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
    def trainer(self):
        return {key: copy.deepcopy(self.__getattribute__(key))
                for key in ('model_path', 'device', 'xtype', 'ytype', 'model_name')}

    @property
    def fit(self):
        return {key: copy.deepcopy(self.__getattribute__(key)) 
                for key in ('load_model', 'save_model', 'verbose')}

    @property
    def evaluate(self):
        return {key: copy.deepcopy(self.__getattribute__(key)) 
                for key in ('load_model', 'save_metrics', 'verbose')}

    def __getitem__(self, index):
        return self.__getattribute__(index)

    def __len__(self):
        return len(self.__dict__)

    def __repr__(self):
        delimiter = ",\n" + " " * 8
        return "HParams({})".format(delimiter.join("{}={}".format(key, value.__repr__())
            for key, value in self.__dict__.items() if value))

    @classmethod
    def from_parser(cls, parser: ArgumentParser):
        return cls(**vars(parser.parse_args()))

