import copy, torch, argparse

class HParams(object):

    
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default='')
    parser.add_argument('--batch_size', type=int, help="Batch Size in Training", default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of Epochs")
    parser.add_argument('--learn_rate', type=float, default=1e-4, help="Learning Rate")
    parser.add_argument('--lr_decay', type=float, default=0.96, help="Learning Rate Decay")
    parser.add_argument('--weight_decay', type=float, default=0.0000, help="Weight Decay")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'both'])
    parser.add_argument('--model_path', type=str, default='checkpoints')
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--load_model', type=int, default=False)
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--normalize', action='store_true', help="Normalize input data")
    #parser.add_argument('--wandb', action='store_true', help='Toggle for Weights & Biases (wandb)')
    #parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                                        help='The device to run on models, cuda is default.')
    '''


    """Hyperparameters used for training."""
    def __init__(self, parser=None, *args, **kwargs):

        if parser is None:
            parser = self.parser

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
        
        self.update(**vars(parser.parse_args()))
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
    def from_parser(cls, parser: argparse.ArgumentParser):
        return cls(**vars(parser.parse_args()))

