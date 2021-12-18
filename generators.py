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
    def __init__(self, parent=None, folder=None):
        if parent is None:
            parent = self.parent_dir

        if folder:
            folder = os.path.join(parent, folder)
        else:
            folder = parent

        makedirs(folder)
        self.parent = parent
        self.folder = folder

    @property
    def path(self):
        return os.path.normpath(os.path.join(self.parent, self.folder))


class FileGenerator(Generator):
    def __init__(self, buf, parent=None, folder=None):
        super(FileGenerator, self).__init__(parent=parent, folder=folder)
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
    def __init__(self, buf, path=None, project=None, entity=None, model=None, **config):
        if not buf.endswith('.tex'):
            buf = buf + '.tex'

        
        self.fname = buf.split('.tex')[0]
        self._instances.append(self)

        super(LatexGenerator, self).__init__(buf, parent=path, folder='latex')
        
        self._buffer.write(r'\documentclass{article}')
        self._buffer.write(r'\usepackage[utf8]{inputenc}')
        self._buffer.write(r'\usepackage{booktabs}')
        
        self._buffer.write(r'\title{' + str(project) + '}')
        self._buffer.write(r'\author{' + str(entity) + '}')
        self._buffer.write(r'\date{December 2021}')

        self._buffer.write(r'\begin{document}')
        self._buffer.write(r'\maketitle')

    def __del__(self):
        self._buffer.write(r'\end{document}')
        super().__del__()

    def create_main_page(self):
        with open(os.path.join(self.folder, self.main_page), 'w') as f:
            for each in self._instances:
                f.write(f'\\include{each.fname}')

    def __call__(self, *dfs):
        for df in dfs:
            df.to_html(self._buffer)

    def add_image(self, path):
       self._buffer.write(r"\begin{figure}\includegraphics[width=\linewidth]{" + path \
                + r"}\caption{A boat.}\label{fig:boat1}\end{figure}")


class HTMLGenerator(FileGenerator):
    main_page = 'index.html'
    _instances = []
    from dominate import tags

    def __init__(self, buf, path=None, project=None, entity=None, model=None, **config):
        if not buf.endswith('.html'):
            buf = buf + '.html'

        super(HTMLGenerator, self).__init__(buf, parent=path, folder='html')
        
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

    def add_figure(self, fig):
        path = os.path.join(self.path, "figure.png")
        i = 0
        while os.path.isfile(path):
            path = os.path.join(self.path, f"figure{i}.png")
            i += 1
        fig.plot(path=path)
        self._buffer.write(str(self.tags.img(href=path)))


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
        super(FigureGenerator, self).__init__(parent=path, folder='plot')
        
        if fig is None:
            fig = plt.figure()

        axes = fig.subplots(*axes, **kwargs)
        
        if isinstance(axes, plt.Axes):
            axes = np.array([axes])

        self.fig = fig
        self.axes = axes

    def __getitem__(self, index):
        return self.axes[index]

    def __len__(self):
        return len(self.axes)

    def __setitem__(self, index, axis):
        self.axes[index] = axis(self.axes[index])

    def plot(self, fname=None, path=None, *argv, **kwargs):
        """
                f_none f_exist
        p_none  show    save(s.p, f)
        p_exist save(p) save(p, f)
        """
        if fname is None and path is None:
            plt.show()
        else:
            if path is None:
                path = self.path
            if fname is not None:
                path = os.path.normpath(os.path.join(path, fname))
            plt.savefig(path)

    def __del__(self):
        plt.close(self.fig)

    class Semilogy(_AxisGenerator):
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
