from .dataset import Dataset


def mean_std_norm(dataset: Dataset):
    dataset.features -= dataset.features.mean()
    dataset.features /= dataset.features.std()

    dataset.labels -= dataset.labels.mean()
    dataset.labels /= dataset.labels.std()

def l1_norm(dataset: Dataset):
    dataset.features /= dataset.features.sum()
    dataset.labels /= dataset.labels.sum()

def l2_norm(dataset: Dataset):
    dataset.features /= dataset.features.square().sum()
    dataset.labels /= dataset.labels.square().sum()

def min_max_norm(dataset: Dataset):
    fmin = dataset.features.min()
    fmax = dataset.features.max()
    lmin = dataset.labels.min()
    lmax = dataset.labels.max()
    dataset.features -= fmin
    dataset.features /= fmax - fmin

    dataset.labels -= lmin
    dataset.labels /= lmax - lmin
    
    return fmin, fmax, lmin, lmax