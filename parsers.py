import datetime
import pandas as pd


class Parser:
    def __init__(self, buf):
        self._buffer = open(buf, 'w')

    def __del__(self):
        self._buffer.close()


class WandbParser:
    _instances = []
    def __init__(self, project=None, entity=None, model=None, **config):
        self.wandb = wandb.init(project=project, entity=entity, config=config)
        
        if model is not None:
            self.wandb.watch(self.model, log="all")

    def __call__(self, train_df=None, valid_df=None, test_df=None):
        pass

class LatexParser(Parser):
    main_page = 'index.html'
    _instances = []
    def __init__(self, buf, project=None, entity=None, model=None, **config):
        super(LatexParser, self).__init__(buf)
        
        self._buffer.write('\\documentclass{article}\n')
        self._buffer.write('\\usepackage[utf8]{inputenc}\n')
        self._buffer.write('\\usepackage{booktabs}\n')
        
        self._buffer.write('\\title{' + str(project) + '}\n')
        self._buffer.write('\\author{' + str(entity) + '}\n')
        self._buffer.write('\\date{December 2021}\n')

        self._buffer.write('\\begin{document}\n')
        self._buffer.write('\\maketitle\n')

    def __del__(self):
        self._buffer.write('\\end{document}\n')
        super().__del__()

    @classmethod
    def create_main_page(cls):
        with open(cls.main_page, 'w') as f:
            for each in cls._instances:
                f.write(f'\\include{each.fname}')

    def __call__(self, train_df=None, valid_df=None, test_df=None):
        
        if train_df is not None:
            train_df.to_latex(self._buffer)

        if valid_df is not None:
            train_df.to_latex(self._buffer)

        if test_df is not None:
            train_df.to_latex(self._buffer)
           

class HTMLParser(Parser):
    main_page = 'index.html'
    _instances = []

    def __init__(self, buf, project=None, entity=None, model=None, **config):
        super(HTMLParser, self).__init__(buf)
        self.fname = buf
        self._instances.append(self)

    @classmethod
    def create_main_page(cls):
        with open(cls.main_page, 'w') as f:
            for each in cls._instances:
                f.write(f'<a href="{each.fname}">{each.fname.split(".")[0]}</a><br>')

    def __call__(self, train_df=None, valid_df=None, test_df=None):
        self._buffer.write(f'<a href="{self.main_page}">go back</a><br>')
        
        if train_df is not None:
            train_df.to_html(self._buffer)

        if valid_df is not None:
            train_df.to_html(self._buffer)

        if test_df is not None:
            train_df.to_html(self._buffer)
