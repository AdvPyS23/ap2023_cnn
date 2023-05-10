"""Main python script, used to build the model from scratch
"""


import scripts.preprocessing as preprocessing


# Load the data, transform it
train_data, test_data, train_loader, test_loader = preprocessing.load(32, 1)


# Checking dataset size and visualizing a few images
preprocessing.data_check(train_data, test_data, train_loader)
