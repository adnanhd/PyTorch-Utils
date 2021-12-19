#!/usr/bin/env python3
import torch, os, json

import utils.metrics
import utils.trainer
import utils.params

import utils.data
import utils.models
#import utils.losses
import utils.callbacks
import utils.generators

epsilon = 1e-5
torch.backends.cudnn.benchmark = True


class PyTorchUtils(object):
    def __init__(self, experiment="first_entity", description=None):
        self.parser = utils.params.argparse.ArgumentParser(
            description=description)
        self.trainer = None
        self.metrics = dict()
        self.parser.add_argument('--batch_size', type=int, help="Batch Size in Training", default=16)
        self.parser.add_argument('--num_workers', type=int, default=0)
        self.parser.add_argument('--num_epochs', type=int, default=100, help="Number of Epochs")
        self.parser.add_argument('--learn_rate', type=float, default=1e-4, help="Learning Rate")
        self.parser.add_argument('--lr_decay', type=float, default=0.96, help="Learning Rate Decay")
        self.parser.add_argument('--weight_decay', type=float, default=0.0000, help="Weight Decay")
        self.parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'both'])
        self.parser.add_argument('--save_path', type=str, default='checkpoints')
        self.parser.add_argument('--save_model', type=bool, default=False)
        self.parser.add_argument('--load_model', type=int, default=None)
        self.parser.add_argument('--multi_gpu', type=bool, default=False)
        self.parser.add_argument('--normalize', action='store_true', help="Normalize input data")
        self.parser.add_argument('--wandb', action='store_true', help='Toggle for Weights & Biases (wandb)')
        self.parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                                            help='The device to run on models, cuda is default.')
        self.parse_args()
        self.generator = utils.generators.HTMLGenerator(
            project=self.project, entity=experiment, 
            main_page=True)
        self.experiment = experiment

    @property
    def project(self):
        return self.hparams.save_path

    def parse_args(self):
        self.hparams = self.parser.parse_args()

    def garbage(self):
        if self.hparams.normalize:
            inp = dataset.features.float()

            mean = torch.mean(inp, dim=0) + epsilon
            std = torch.std(inp, dim=0) + epsilon

            dataset.transform = transforms.Compose(
                [transforms.Normalize(mean=mean, std=std)])

    def compile(self, model, loss, metrics):
        device = torch.device(
            self.hparams.device if torch.cuda.is_available() else 'cpu')

        self.trainer = utils.trainer.Trainer(model=model, 
                            save_path=self.hparams.save_path,
                               device=device, xtype=torch.float, ytype=torch.float)

        self.trainer.loss_func = loss

        self.trainer.optimizer = torch.optim.Adam(self.trainer.model.parameters(),
                                lr=self.hparams.learn_rate, weight_decay=self.hparams.weight_decay)

        self.metrics.update(metrics)
        self.generator.doc.body.add(self.generator.tags.h2("Compile"))
        self.generator(self.trainer.model, self.trainer.loss_func)

    def train(self, train_dataset, valid_dataset=None, callbacks=[], **kwargs):
        fig = utils.generators.FigureGenerator(1, project=self.project)

        train_loader = train_dataset.dataloader(train=True,
                                                batch_size=self.hparams.batch_size,
                                                num_workers=self.hparams.num_workers)

        if valid_dataset is None:
            valid_loader = None
        else:
            valid_loader = valid_dataset.dataloader(
                batch_size=1, train=False,
                num_workers=self.hparams.num_workers)

        df = self.trainer.fit(epochs=self.hparams.num_epochs,
                    train_dataset=train_loader,
                    valid_dataset=valid_loader,
                    load_model=self.hparams.load_model,
                    save_model=self.hparams.save_model,  # save model and losses
                    callbacks=callbacks,
                    metrics=self.metrics,
                    verbose=True, **kwargs)  # print and save logs

        fig[0] = utils.generators.FigureGenerator.Semilogy(df)
        self.generator.doc.body.add(self.generator.tags.h2("Training"))
        fname = self.generator.fname.split('.')[0] + '.png'
        self.generator.add_figure(fig, fname='train' + fname)
        self.generator(df)
        return df

    def test(self, test_dataset, callbacks=[]):
        fig = utils.generators.FigureGenerator(1, project=self.project)

        test_loader = test_dataset.dataloader(
            batch_size=1, train=False,
            num_workers=self.hparams.num_workers)

        df = self.trainer.evaluate(
                test_dataset=test_loader,
                load_model=self.hparams.save_model,
                callbacks=callbacks,
                metrics=self.metrics,
                verbose=True)  # print and save logs

        fig[0] = utils.generators.FigureGenerator.Histogram(df)
        self.generator.doc.body.add(self.generator.tags.h2("Testing"))
        fname = self.generator.fname.split('.')[0] + '.png'
        self.generator.add_figure(fig, fname='test' + fname)
        self.generator(df)
        return df

    def log(self, fname=None): # 
        if fname is None:
            fname = os.path.join(self.project, f".{self.experiment}_runtimeinfo.log")

        with open(fname, 'w') as text_file:
            print('Memory Loc.:  ', self.trainer.device, file=text_file)
            #print('Device Name:  ', torch.cuda.get_device_name(0), file=text_file)

            print('Allocated:    ', round(torch.cuda.memory_allocated(
                0)/1024**2, 1), 'MB', file=text_file)
            print('Cached:       ', round(torch.cuda.memory_reserved(
                0)/1024**2, 1), 'MB', file=text_file)

        
        if self.hparams.mode in ('train', 'both'):        
            with open(os.path.join(self.project, f".{self.experiment}_hparams.json") ,'w') as j:
                json.dump(vars(self.hparams), j, indent=2, sort_keys=True) 
