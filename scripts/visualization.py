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

def loss_acc_plot(train_loss, test_loss, train_acc, test_acc, epochs):
    """Plots the change of loss and accuracy through the training process

    Parameters
    -----------
    train_loss
        List containing the average loss per hundred batches through all epochs
    test_loss
        List containing the average loss per * batches through all epochs
    train_acc
        List containing the average accuracy per 100 batches for training and testing
    test_acc
        List containing the average accuracy per 100 batches for training and testing
    epochs
        Number of epochs the model was trained for
    """
    epochs = 20
    steps_train = np.linspace(1, epochs + 1, 18 * epochs)
    #steps_test = np.linspace(1, epochs + 1, ?)
    # Initiate plotting
    fig, axs = plt.subplots(1, 2)
    axs[0, 0].plot(steps_train, train_loss, '-o')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss')

    axs[0, 1].plot(steps_train, train_acc, '-o')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('Accuracy (%)')

    plt.savefig('./data/acc_loss.png')
