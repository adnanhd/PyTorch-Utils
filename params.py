import torch
from typing import Optional, Union
from argparse import ArgumentParser
from dataclasses import dataclass, field


@dataclass
class HParams(object):

    model_name: str = field(
        metadata={"help": "model's name to name the containing model's weights"}
    )

    model_path: Optional[str] = field(
        default="checkpoints",
        metadata={
            "help": "model's path to save and load the containing model's weights"}
    )

    device: torch.device = field(
        default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        metadata={"help": "medium device to host the model"}
    )

    xtype: torch.dtype = field(
        default=torch.float, metadata={"help": "dtpye of model's inputs"}
    )

    ytype: torch.dtype = field(
        default=torch.float, metadata={"help": "dtpye of model's outputs"}
    )


@dataclass
class TrainerArguments:

    experiment: str = field(
        metadata={"help": "the test condition's name to discriminate from others"}
    )

    verbose: bool = field(
        default=True, metadata={'help': "prints metrics during training and testing if is set"}
    )

    # training parameters
    num_epochs: int = field(
        default=100, metadata={'help': "number of epochs to use for training"}
    )

    batch_size: int = field(
        default=16, metadata={'help': "batch size to pack dataset used in training"}
    )

    learn_rate: int = field(
        default=1e-4, metadata={'help': "learning rate of the model used in training"}
    )

    lr_decay: Optional[int] = field(
        default=0.96, metadata={'help': "learning rate decay"}
    )

    weight_decay: Optional[int] = field(
        default=0.0000, metadata={'help': "weight decay"}
    )

    save_metrics: bool = field(
        default=True, metadata={"help": "stores calculated metrics if is set"}
    )

    save_model: bool = field(
        default=True, metadata={"help": "saves the model to the store path at the end of testing"}
    )

    load_model: Union[str, bool] = field(
        default=False, metadata={"help": "path to load model for testing"}
    )
