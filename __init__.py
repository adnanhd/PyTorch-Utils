#!/usr/bin/env python3
import torch, os, json

import utils.metrics
import utils.trainer
import utils.params
import utils.config

import utils.data
import utils.models
import utils.callbacks
import utils.generators

epsilon = 1e-5
torch.backends.cudnn.benchmark = True


class Pipeline(object):
    def __init__(self, model_name=None, experiment="first_entity", description=None, cfg=utils.params.HParams(), **kwargs):
        self.trainer = None
        self.metrics = dict()
        self.hparams = cfg
        self.hparams.update(**kwargs)
        self.generator = utils.generators.HTMLGenerator(
            project=self.project, entity=experiment, 
            main_page=True)
        self.experiment = experiment

    @property
    def project(self):
        return self.hparams.save_path

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
        self.generator.doc.body.add(self.generator.tags.h2("Hyper Parameters"))
        self.generator.doc.body.add(self.generotor.tags.pre(str(self.hparams)))
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
                streamed_hparams = {key: str(value) for key, value in vars(self.hparams).items()}
                json.dump(streamed_hparams, j, indent=2, sort_keys=True) 
