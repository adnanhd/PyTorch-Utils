import math, os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def makedirs(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

class Generator:
    parent_dir = './'
    def __init__(self, folder=None):
        if folder:
            folder = os.path.join(self.parent_dir, folder)
        else:
            folder = self.parent_dir

        makedirs(folder)
        self.folder = folder


class FileGenerator(Generator):
    def __init__(self, buf, folder=None):
        super(FileGenerator, self).__init__(folder=folder)
        self._buffer = open(os.path.join(self.folder, buf), 'w')
    
    def create_main_page(self):
        return NotImplementedError()

    def __del__(self):
        self._buffer.close()


class WandbGenerator(Generator):
    def __init__(self, project=None, entity=None, model=None, **config):
        self.wandb = wandb.init(project=project, entity=entity, config=config)
        
        if model is not None:
            self.wandb.watch(self.model, log="all")

    def __call__(self, train_df=None, valid_df=None, test_df=None):
        pass

class LatexGenerator(FileGenerator):
    main_page = 'index.tex'
    _instances = []
    def __init__(self, buf, project=None, entity=None, model=None, **config):
        if not buf.endswith('.tex'):
            buf = buf + '.tex'

        
        self.fname = buf.split('.tex')[0]
        self._instances.append(self)

        super(LatexGenerator, self).__init__(buf, folder='latex')
        
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

    def create_main_page(self):
        with open(os.path.join(self.folder, self.main_page), 'w') as f:
            for each in self._instances:
                f.write(f'\\include{each.fname}')

    def __call__(self, *dfs):
        for df in dfs:
            df.to_html(self._buffer)
           

class HTMLGenerator(FileGenerator):
    main_page = 'index.html'
    _instances = []
    from dominate import tags

    def __init__(self, buf, project=None, entity=None, model=None, **config):
        if not buf.endswith('.html'):
            buf = buf + '.html'

        super(HTMLGenerator, self).__init__(buf, folder='html')
        
        self.fname = buf.split('.html')[0]
        self._instances.append(self)
        self.basecontent = config

    def create_main_page(self):
        with open(os.path.join(self.folder, self.main_page), 'w') as f:
            for each in self._instances:
                f.write(str(self.tags.a(each.fname, href=each.fname + '.html')) + str(self.tags.br()))

    def __call__(self, *dfs):
        self._buffer.write(str(self.tags.a('go back', href=self.main_page)) + str(self.tags.br()))
        
        for df in dfs:
            df.to_html(self._buffer)


class _AxisGenerator(Generator):
    def __init__(self, title=None, ylabel=None, xlabel=None):
        self.title = title
        self.ylabel = ylabel
        self.xlabel = xlabel

    def __call__(self, ax=None):
        if ax is None:
            ax = plt.axis()
        if self.title is not None:
            ax.set_title(self.title)
        if self.xlabel is not None:
            ax.set_xlabel(self.xlabel)
        if self.ylabel is not None:
            ax.set_ylabel(self.ylabel)
        return ax


class FigureGenerator(Generator):
    def __init__(self, *axes, fig=None, path=None, **kwargs):
        super(FigureGenerator, self).__init__(folder='plot')
        
        if fig is None:
            fig = plt.figure()

        axes = fig.subplots(*axes, **kwargs)
        
        if isinstance(axes, plt.Axes):
            axes = np.array([axes])

        if path is not None and os.path.isdir(path):
            path = os.path.join(path, 'figure.png')

        self.fig = fig
        self.axes = axes
        self.path = os.path.join(self.folder, path)

    def __getitem__(self, index):
        return self.axes[index]

    def __len__(self):
        return len(self.axes)

    def __setitem__(self, index, axis):
        self.axes[index] = axis(self.axes[index])

    def plot(self, *argv, **kwargs):
        if self.path is None:
            plt.show()
        else:
            plt.savefig(self.path)

    def __del__(self):
        plt.close(self.fig)

    class Semilogy(_AxisGenerator):
        default_path = 'semilogy.png'
        def __init__(self, df, title=None, xlabel=None, ylabel=None):
            self.df = df
            
            if title is None:
                title=f'Training and Validation Metrics in {len(df)} Iter'
            if xlabel is None:
                xlabel='Epochs'
            if ylabel is None:
                ylabel='Loss'

            super(self.__class__, self).__init__(title=title, ylabel=ylabel, xlabel=xlabel)
            
        def __call__(self, ax):
            # TODO: add 'x' and 'o' for train and valid
            super().__call__(ax=ax)
            ax.semilogy(self.df, 'o-')
            ax.legend(self.df.columns)
            return ax
    
    class Histogram(_AxisGenerator):
        default_path = 'histogram.png'
        def __init__(self, df, ax=None, title=None, xlabel=None, ylabel=None):
            self.df = df
            
            if title is None:
                title=f'Testing Metrics in {len(df)} Iter'
            if xlabel is None:
                xlabel='Metrics'
            if ylabel is None:
                ylabel='Cases'

            super(self.__class__, self).__init__(title=title, ylabel=ylabel, xlabel=xlabel)

        def __call__(self, ax):
            # TODO: add 'x' and 'o' for train and valid
            super().__call__(ax=ax)
            ax.hist(self.df, bins=int(math.log2(len(self.df) ** 2)) if len(self.df) else None)
            ax.legend(self.df.columns)
            return ax
