"""Main python script, used to build the model from scratch
"""


import scripts.visualization as visu
import scripts.preprocessing as preprocessing
import matplotlib.pyplot as plt
import numpy as np
from torchvision import utils

# Load the data, transform it
train_data, test_data, train_loader, test_loader = preprocessing.load(32, 1)


# Visualizing the data (will move this to the appropriate scripts over the weekend)
# --------------------

def imsave(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('./data/batchtest.jpg')          # Had to save the image instead of imshow due to pycharm graphics problem

dataiter = iter(training_loader)
images, labels = next(dataiter)

imsave(utils.make_grid(images))

print(f'See ./data/batchtest.jpg to visualize a batch of training images')
