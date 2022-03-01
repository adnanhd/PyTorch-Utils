import os, tqdm, hashlib, torch, numpy as np
from .dataset import Dataset, generate_dataset
from sklearn.model_selection import train_test_split


def _get_data(feature_class, label_class, path=None, 
              cache=True, transform=None, normalize=False,
              feature_func=None, label_func=None, device='cuda'):
    if not isinstance(device, torch.device):
        device = torch.device(device)

    #cache_filepath = hashlib.sha256(dataset_name.encode('utf-8')).hexdigest()
    cache_filepath = hex(int(label_class.hash(), 16) ^ int(feature_class.hash(), 16))[2:]
    cache_filepath = os.path.join(path, cache_filepath)

    if cache and os.path.isfile(cache_filepath):
        data = torch.load(cache_filepath, map_location=device)
        features, labels = data['features'], data['labels']
    else:
        data = tuple(generate_dataset(feature_class, label_class, filepath=path))
            
        features = torch.empty(torch.Size([len(data), *feature_class.shape]), device=device)
        labels = torch.empty(torch.Size([len(data), *label_class.shape]), device=device)
            
        for i, dname in enumerate(tqdm.tqdm(data)):
            features[i] = feature_class.load(path, dname).data
            labels[i] = label_class.load(path, dname).data
            #fnameDict[features[i].cpu().numpy().tobytes()] = dname

        if cache:
            torch.save({'features': features, 'labels': labels}, cache_filepath)

        if normalize:
            features = torch.nn.functional.normalize(features, dim=(0, 1))
            
    return features, labels


def get_dataset(feature_class, label_class, path=None, 
                cache=True, transform=None, normalize=False,
                feature_func=None, label_func=None, device='cuda'):
    features, labels = _get_data(feature_class, label_class, path=path,
                                 cache=cache, transform=transform, 
                                 normalize=normalize, device=device)
    
    return Dataset(features=features, labels=labels, transform=transform, 
                   feature_func=feature_func, label_func=label_func)


def get_train_test_datasets(feature_class, label_class, path=None, 
                cache=True, transform=None, normalize=False, test_size=10,
                feature_func=None, label_func=None, device='cuda'):
    features, labels = _get_data(feature_class, label_class, path=path,
                                 cache=cache, transform=transform, 
                                 normalize=normalize, device=device)
    
    data = train_test_split(features, labels, test_size=test_size, shuffle=True)
    
    train_dataset = Dataset(features=data[0], labels=data[1], 
            feature_func=feature_func, label_func=label_func,
            transform=transform)

    valid_dataset = Dataset(features=data[2], labels=data[3], 
            feature_func=feature_func, label_func=label_func,
            transform=transform)

    return train_dataset, valid_dataset
