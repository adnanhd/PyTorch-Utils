import os, math
import matplotlib.pyplot as plt


def _subplots(ylabel=None, xlabel=None, title=None, ax=None, fig=None):
    if not ax:
        fig, ax = plt.subplots()
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)

    return fig, ax

def semilogy(df, path=None, ax=None, fig=None, 
        title=None, xlabel=None, ylabel=None, **kwargs):
    if title is None:
        title=f'Training and Validation Metrics in {len(df)} Iter'
    if xlabel is None:
        xlabel='Epochs'
    if ylabel is None:
        ylabel='Loss'
    fig, ax = _subplots(title=title, xlabel=xlabel, ylabel=ylabel, fig=fig, ax=ax)

    # TODO: add 'x' and 'o' for train and valid
    ax.semilogy(df, 'o-')
    ax.legend(df.columns)
    if path is not None:
        if os.path.isdir(path):
            path = os.path.join(path, f'logy_{len(df)}_iter.png')
        plt.savefig(path)
        plt.close(fig)
    return fix, ax

def histogram(df, path=None, ax=None, fig=None, 
        title=None, xlabel=None, ylabel=None, **kwargs):
    if title is None:
        title=f'Testing Metrics in {len(df)} Iter'
    if xlabel is None:
        xlabel='Metrics'
    if ylabel is None:
        ylabel='Cases'
    fig, ax = _subplots(title=title, xlabel=xlabel, ylabel=ylabel, fig=fig, ax=ax)

    # TODO: add 'x' and 'o' for train and valid
    ax.hist(df, bins=int(math.log2(len(df) ** 2)))
    ax.legend(df.columns)
    if path is not None:
        if os.path.isdir(path):
            path = os.path.join(path, f'hist_{len(df)}_iter.png')
        plt.savefig(path)
        plt.close(fig)
    return fig, ax
