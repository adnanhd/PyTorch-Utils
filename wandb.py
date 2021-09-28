import neural_structured_learning as nsl

class HParams(object):
  """Hyperparameters used for training."""
  def __init__(self, learn_rate=4e-5, batch_size=1, epochs=1):
    ### wandb config
    self.config = {
        "project": "graph-neural-network",
        "entity": "adnanhd",
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": learn_rate
    }
    
    ### training parameters
    self.train_epochs = 50 #500
    self.batch_size = 128
    self.learn_rate = 1e-5
    
    ### evaluating parameters
    self.eval_steps = None  # All instances in the test set are evaluated.

    @property
    def epochs(self):
        return self.config['epochs']
    
    @epochs.setter
    def epochs(self, new_epochs):
        self.config['epochs'] = new_epochs

    @property
    def batch_size(self):
        return self.config['batch_size']
    
    @batch_size.setter
    def epochs(self, new_size):
        self.config['batch_size'] = new_size

    @property
    def learn_rate(self):
        return self.config['epochs']
    
    @epochs.setter
    def epochs(self, new_epochs):
        self.config['epochs'] = new_epochs
