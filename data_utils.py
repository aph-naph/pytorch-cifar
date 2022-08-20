import torch
import numpy as np
import random

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def get_data_loaders(dataset,
                     data_dir,
                     train_transform,
                     val_transform,
                     test_transform,
                     batch_size,
                     val_split=0.1,
                     random_seed=0,
                     num_workers=4):
    
    train_loader, val_loader = get_train_val_loaders(
        dataset, data_dir, train_transform, val_transform, batch_size,
        val_split=val_split, random_seed=random_seed, num_workers=num_workers
    )

    test_loader = get_test_loader(
        dataset, data_dir, test_transform, batch_size,
        random_seed=random_seed, num_workers=num_workers
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
                      num_workers=4):

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
        trainset, sampler=SubsetRandomSampler(train_indices),
        batch_size=batch_size, num_workers=num_workers)

    val_loader = torch.utils.data.DataLoader(
        valset, sampler=SubsetRandomSampler(val_indices),
        batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader


def get_test_loader(dataset,
                    data_dir,
                    test_transform,
                    batch_size,
                    random_seed=0,
                    num_workers=4):

    testset = dataset(root=data_dir, train=False, download=True, transform=test_transform)

    return torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=4)
