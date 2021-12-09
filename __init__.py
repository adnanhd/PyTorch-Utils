#!/usr/bin/env python3

import utils.metrics
import utils.trainer
import utils.params

import utils.data
import utils.models
#import utils.losses
import utils.callbacks
import utils.parsers
import utils.plots

class PyTorchUtils:
    def __init__(self, description=None):
        self.parser = utils.params.argparse.ArgumentParser(description=description)

        self.parser.add_argument('--batch_size', type=int, help="Batch Size in Training", default=16)
        self.parser.add_argument('--num_workers', type=int, default=0)
        self.parser.add_argument('--num_epochs', type=int, default=100, help="Number of Epochs")
        self.parser.add_argument('--learn_rate', type=float, default=1e-4, help="Learning Rate")
        self.parser.add_argument('--lr_decay', type=float, default=0.96, help="Learning Rate Decay")
        self.parser.add_argument('--weight_decay', type=float, default=0.0000, help="Weight Decay")
        self.parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'both'])
        self.parser.add_argument('--save_path', type=str, default='./model')
        self.parser.add_argument('--print_iters', type=int, default=10)
        self.parser.add_argument('--save_iters', type=int, default=50)
        self.parser.add_argument('--test_iters', type=int, default=100)
        self.parser.add_argument('--restart_iters', type=int, default=0)
        self.parser.add_argument('--multi_gpu', type=bool, default=False)
        self.parser.add_argument('--normalize', action='store_true', help="Normalize input data")
        self.parser.add_argument('--wandb', action='store_true', help='Toggle for Weights & Biases (wandb)')
        self.parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], 
                                        help='The device to run on models, cuda is default.')

    def main(self):
        pass
