from argparse import ArgumentParser

class CnnFoilConfig:
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, help="Batch Size in Training", default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of Epochs")
    parser.add_argument('--learn_rate', type=float, default=1e-4, help="Learning Rate")
    parser.add_argument('--lr_decay', type=float, default=0.96, help="Learning Rate Decay")
    parser.add_argument('--weight_decay', type=float, default=0.0000, help="Weight Decay")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'both'])
    parser.add_argument('--save_path', type=str, default='checkpoints')
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--load_model', type=int, default=None)
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--normalize', action='store_true', help="Normalize input data")
    #parser.add_argument('--wandb', action='store_true', help='Toggle for Weights & Biases (wandb)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                                        help='The device to run on models, cuda is default.')

class Config:
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='network')
    parser.add_argument('--model_path', type=str, default='checkpoints')
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--load_model', type=int, default=False)

    parser.add_argument('--num_train_epochs', type=int, default=100, help="Number of Epochs")
    parser.add_argument('--train_batch_size', type=int, help="Batch Size in Training", default=16)
    parser.add_argument('--eval_batch_size', type=int, help="Batch Size in Training", default=None)
    parser.add_argument('--logging_steps', type=int, default=100)

    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning Rate")

    parser.add_argument('--lr_decay', type=float, default=0.96, help="Learning Rate Decay")
    parser.add_argument('--weight_decay', type=float, default=0.0000, help="Weight Decay")

    parser.add_argument('--do_train', default=False, action='store_true')
    parser.add_argument('--do_test', default=False, action='store_true')

    parser.add_argument('--multi_gpu', action='store_true', default=False)
    parser.add_argument('--normalize', action='store_true', help="Normalize input data")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                                        help='The device to run on models, cuda is default.')

    @classmethod
    def trainer_arguments(cls):
        pass

    @classmethod
    def model_arguments(cls):
        pass
 
