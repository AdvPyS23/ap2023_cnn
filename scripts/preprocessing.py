"""Contains all data loading and preprocessing tasks

This script contains the functions used for the two main preprocessing tasks:
the loading/transformation of data and creation of dataloaders,
and the visualization of the input images
"""
from torch.utils.data import DataLoader
from torchvision import utils
from torchvision import datasets
import torchvision.transforms as transforms


def load(batch_size, cores):
    """Loads the Fashion-MNIST dataset, normalizes it and creates the DataLoaders.

    The Fashion-MNIST dataset is available directly through Torchvision.

    Parameters
    -----------
    batch_size
        Size of the batches loaded into the
    cores
        Number of CPUs available
    Returns
    ---------
    tuple
        A tuple (training_data, test_data, training_loader, test_loader), the Dataset and DataLoader objects for the
        training and test sets.

    """

    training_data = datasets.FashionMNIST(
        root="../data", train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    )

    test_data = datasets.FashionMNIST(
        root="../data", train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # ToTensor scales to [0,1]. This centers data around 0
        ])
    )

    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=cores)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=cores)

    return training_data, test_data, training_loader, test_loader


def data_check(training_data, test_data, training_loader, test_loader):
    """Prints dataset and batch size, plots a batch of training images.

    Parameters
    ----------
    training_data
        Dataset object for the training data
    test_data
        Dataset object for the test data
    training_loader
        DataLoader for the training data
    test_loader
        DataLoader for the test data
    """
    print(f'Loaded {len(training_data)} training images and {len(test_data)} test images\n')

