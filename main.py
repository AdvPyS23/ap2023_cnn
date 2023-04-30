# Loading the dataset
# ---------------------
#
# The Fashion-MNIST dataset is available directly through Torchvision
# It contains 60'000 training images and 10'000 test images


from torch.utils.data import DataLoader
from torchvision import utils
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


training_data = datasets.FashionMNIST(
    root="data", train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
)

test_data = datasets.FashionMNIST(
    root="data", train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # ToTensor scales to [0,1]. This centers data around 0
    ])
)

print(f'Loaded {len(training_data)} training images\nand {len(test_data)} test images')

# DataLoaders

batchsize = 32

training_loader = DataLoader(training_data, batch_size=batchsize, shuffle=True) #, num_workers=4)
test_loader = DataLoader(test_data, batch_size=batchsize, shuffle=True) #, num_workers=4)

print(f'Datasets and DataLoaders ready, batch size: {batchsize}')

# Visualizing the data
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
