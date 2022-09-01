import torch
import numpy as np
import random

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(123)

def get_data_loaders(dataset,
                     data_dir,
                     train_transform,
                     val_transform,
                     test_transform,
                     batch_size,
                     val_split=0.1,
                     random_seed=0,
                     num_workers=4,
                     worker_init_fn=None):
    
    train_loader, val_loader = get_train_val_loaders(
        dataset, data_dir, train_transform, val_transform, batch_size,
        val_split=val_split, random_seed=random_seed, num_workers=num_workers,
        worker_init_fn=worker_init_fn
    )

    test_loader = get_test_loader(
        dataset, data_dir, test_transform, batch_size,
        random_seed=random_seed, num_workers=num_workers,
        worker_init_fn=worker_init_fn
    )

    return train_loader, val_loader, test_loader

def get_train_val_loaders(dataset,
                      data_dir,
                      train_transform,
                      val_transform,
                      batch_size,
                      val_split=0.1,
                      shuffle=True,
                      random_seed=0,
                      num_workers=4,
                      worker_init_fn=None):

    np.random.seed(0)
    trainset = dataset(root=data_dir, train=True, download=True, transform=train_transform)
    valset = dataset(root=data_dir, train=True, download=True, transform=val_transform)

    ds_size = len(trainset)
    indices = np.arange(len(trainset))
    if shuffle:
        np.random.seed(0)
        np.random.shuffle(indices)

    split = int(np.floor(ds_size * val_split))
    train_indices, val_indices = indices[split:], indices[:split]

    train_loader = torch.utils.data.DataLoader(
        trainset, sampler=SubsetRandomSampler(train_indices), batch_size=batch_size,
        num_workers=num_workers, worker_init_fn=worker_init_fn)

    val_loader = torch.utils.data.DataLoader(
        valset, sampler=SubsetRandomSampler(val_indices), batch_size=batch_size,
        num_workers=num_workers, worker_init_fn=worker_init_fn)

    return train_loader, val_loader


def get_test_loader(dataset,
                    data_dir,
                    test_transform,
                    batch_size,
                    random_seed=0,
                    num_workers=4,
                    worker_init_fn=None):

    testset = dataset(root=data_dir, train=False, download=True, transform=test_transform)

    return torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, worker_init_fn=worker_init_fn)
