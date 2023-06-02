# Welcome!
This project is being conducted as part of the **Advanced Python (FS2023)** course at the University of Bern. It aims at building a multi-class classifier for clothing images from the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) using a convolutional neural network built in Pytorch.   
Fashion-MNIST is a dataset of Zalando article photos, with 60,000 examples in the training set and 10,000 examples in the test set. Each sample is a 28x28 grayscale image with a label from one of ten classes.
The classes are the following:

* 0: T-shirt/top
* 1: Trouser
* 2: Pullover
* 3: Dress
* 4: Coat
* 5: Sandal
* 6: Shirt
* 7: Sneaker
* 8: Bag
* 9: Ankle boot



### Virtual Environment / Dependencies

Although all necessary packages can be installed manually, we recommend using [Anaconda](https://www.anaconda.com/download#downloads).   
The [environment file](./environment.yml) lists all packages and their versions. If you already have anaconda installed, simply create a new environment using:      

#### For Linux / WSL2
`conda env create -f environment_lx.yml`  

#### For macOS   
`conda env create -f environment_macOS.yml`   

Check that the environment has been correctly installed with   

`conda env list`

It should be listed under the name `pytorch`.   

### Running

To run our base version of the model, first specify your desired batch size, # of cores and # epochs in the `main.py` file.  
Please note that if you change the base batch size, you also need to change it in `scripts/training.py`. We intend on fixing this issue later.

Once the parameters are set, simply run `main.py`.   
The model is saved in `models`, plots of input images, loss and accuracy are saved in `data`.

If you wish to modify the loss function, optimizer parameters or model architecture, head to `scripts/training.py`


#


Some key resources:

[Basic Pytorch Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

[Detailed breakdown of the CNN architecture](https://cs231n.github.io/convolutional-networks/)

[CNN Model Guide ](https://www.kaggle.com/code/pavansanagapati/a-simple-cnn-model-beginner-guide)






