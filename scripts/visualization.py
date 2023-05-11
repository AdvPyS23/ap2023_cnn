"""Contains the details of the data visualization tasks for all steps of the analysis

"""


import matplotlib.pyplot as plt
import numpy as np
from torchvision import utils


def batch_plot(training_loader, path):
    """Plots a batch of training images and saves it.

    Parameters
    -----------
    training_loader
        DataLoader for the training data
    path
        Path to the batch image output
    """

    def imsave(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.savefig(path)

    dataiter = iter(training_loader)
    images = next(dataiter)[0]

    imsave(utils.make_grid(images))
