#!/usr/bin/env python3
import torch, os, json

import utils.data
import utils.models
import utils.loggers
import utils.callbacks

import utils.metrics
import utils.trainer
import utils.params
import utils.config

epsilon = 1e-5
torch.backends.cudnn.benchmark = True


class Pipeline(object):
    from utils.params import TrainerArguments
    from typing import Optional, Union
    def __init__(self, 
            save_path: Optional[str]="checkpoints",
            description: Optional[str]=None, 
            **kwargs):

        self.trainer = None
        self.metrics = dict()
        self.tparams = self.TrainerArguments(**kwargs)
        self.generator = utils.generators.HTMLGenerator(
            project=save_path, 
            fname=self.tparams.experiment, 
            main_page=True)

    def compile(self, 
            model: torch.nn.Module, 
            loss: torch.nn.Module, 
            metrics: Optional[dict]={},
            model_path: Optional[str]=None,
            **kwargs):

        if model_path is None:
            model_path = self.generator.parent

        hparam = utils.params.HParams(model_path=model_path, **kwargs)
        print(hparam, self.tparams)
        self.trainer = utils.trainer.Trainer(
                model=model, 
                loss=loss, 
                optim=torch.optim.Adam(
                    model.parameters(), 
                    lr=self.tparams.learn_rate, 
                    weight_decay=self.tparams.weight_decay),
                **vars(hparam), **vars(self.tparams))

        self.metrics.update(metrics)
        self.generator.doc.body.add(self.generator.tags.h2("Hyper Parameters"))
        self.generator.doc.body.add(self.generator.tags.pre(str(self.tparams)))
        self.generator.doc.body.add(self.generator.tags.h2("Compile"))
        self.generator(self.trainer.model, self.trainer.loss_func)

    def train(self, train_dataset, valid_dataset=None, callbacks=[], metrics={}, **kwargs):
        fig = utils.generators.FigureGenerator(1, project=self.generator.parent)

        train_loader = train_dataset.dataloader(train=True,
                                                batch_size=self.tparams.batch_size)

        if valid_dataset is None:
            valid_loader = None
        else:
            valid_loader = valid_dataset.dataloader(
                batch_size=1, train=False)

        metrics.update(self.metrics)
        kwargs.update(vars(self.tparams))
        df = self.trainer.fit(epochs=self.tparams.num_epochs,
                    train_dataset=train_loader,
                    valid_dataset=valid_loader,
                    callbacks=callbacks,
                    metrics=metrics, 
                    **kwargs)  # print and save logs

        fig[0] = utils.generators.FigureGenerator.Semilogy(df)
        self.generator.doc.body.add(self.generator.tags.h2("Training"))
        fname = self.generator.fname.split('.')[0] + '.png'
        self.generator.add_figure(fig, fname='train' + fname)
        self.generator(df)
        return df

    def test(self, test_dataset, callbacks=[], metrics={}, **kwargs):
        fig = utils.generators.FigureGenerator(1, project=self.generator.parent)

        test_loader = test_dataset.dataloader(
            batch_size=1, train=False)

        metrics.update(self.metrics)
        kwargs.update(vars(self.tparams))
        df = self.trainer.evaluate(
                test_dataset=test_loader,
                callbacks=callbacks,
                metrics=metrics,
                **kwargs)  # print and save logs

        fig[0] = utils.generators.FigureGenerator.Histogram(df)
        self.generator.doc.body.add(self.generator.tags.h2("Testing"))
        fname = self.generator.fname.split('.')[0] + '.png'
        self.generator.add_figure(fig, fname='test' + fname)
        self.generator(df)
        return df

    def log(self, fname=None): # 
        if fname is None:
            fname = os.path.join(self.generator.parent, f".{self.tparams.experiment}_runtimeinfo.log")

        with open(fname, 'w') as text_file:
            print('Memory Loc.:  ', self.trainer.device, file=text_file)
            #print('Device Name:  ', torch.cuda.get_device_name(0), file=text_file)

            print('Allocated:    ', round(torch.cuda.memory_allocated(
                0)/1024**2, 1), 'MB', file=text_file)
            print('Cached:       ', round(torch.cuda.memory_reserved(
                0)/1024**2, 1), 'MB', file=text_file)

        
        if self.tparams.mode in ('train', 'both'):        
            with open(os.path.join(self.generator.parent, f".{self.tparams.experiment}_hparams.json") ,'w') as j:
                streamed_hparams = {key: str(value) for key, value in vars(self.tparams).items()}
                json.dump(streamed_hparams, j, indent=2, sort_keys=True) 
