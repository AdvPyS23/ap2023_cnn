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
    steps_train = np.linspace(1, epochs + 1, 18 * epochs)
    steps_test = np.linspace(1, epochs + 1, 3 * epochs)

    # Initiate plotting
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    axs[0].plot(steps_train, train_loss, '-', label='training')
    axs[0].plot(steps_test, test_loss, '-', label='testing')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].set_ylim(0, 1.5)
    axs[0].legend(loc='upper right', frameon=False)

    axs[1].plot(steps_train, train_acc, '-', label='training')
    axs[1].plot(steps_test, test_acc, '-')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy (%)')
    axs[1].set_ylim(40, 100)
    axs[1].legend(loc='upper right', frameon=False)

    # Save the figure
    plt.savefig('./data/acc_loss.png')
    print("Image has been saved as ./data/acc_loss.png")