import matplotlib.pyplot as plt
import torch
import os


def subplot_train(path, train_loss, valid_loss, **kwargs):
    fig, ax = subplots()
    epochs = len(train_loss)
    ax.set_title('Training and Validation Loss in {} Iter'.format(epochs))
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    # TODO: add 'x' and 'o' for train and valid
    ax.semilogy(range(epochs), train_loss, label='train')
    ax.semilogy(range(epochs), valid_loss, label='valid')
    ax.legend()
    plt.savefig(os.path.join(path, "train_loss_logy_{}_iter.png".format(epochs)))
    plt.close(fig)


def subplot_test(path, epochs, test_loss, **kwargs):
    fig, ax = plt.subplots()
    ax.set_title('Testing Loss in {} Case'.format(len(test_loss)))
    ax.set_ylabel('Cases')
    ax.set_xlabel('Loss (avg. {:.3e})'.format(torch.as_tensor(test_loss).mean()))
    ax.hist(test_loss, bins=int(math.log2(len(test_loss) ** 2)))
    plt.savefig(os.path.join(path, 'test_loss_hist_{}_iter.png'.format(epochs)))
    plt.close(fig)


def subplots(ylabel=None, xlabel=None, title=None, ax=None, fig=None):
    if not ax:
        fig, ax = plt.subplots()
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)

    return fig, ax

