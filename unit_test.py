import unittest
import scripts.preprocessing as pre


class TestLoad(unittest.TestCase):
    def test_data_loader_size(self):
        batch_size = 32
        cores = 4
        training_data, test_data, training_loader, test_loader = pre.load(batch_size, cores)
        # check if length of training and test loader is equal to the length of training
        # and test data divided by batch size
        # -> ensures that all training and test data is being loaded into
        # the training / test loader with the given batch size
        # without drop_last=True in the DataLoader function, an unexpected output
        # may occur because with the batch size of 32, not all test data can be loaded into
        # the test loader because the data size is not an exact multiple of the batch size
        self.assertEqual(len(training_loader), len(training_data)//batch_size)
        self.assertEqual(len(test_loader), len(test_data)//batch_size)

    def test_data_loader_worker(self):
        batch_size = 32
        cores = 4
        training_data, test_data, training_loader, test_loader = pre.load(batch_size, cores)
        # check if the number of workers for the training and test loaders
        # are the same as the specified number of cores
        self.assertEqual(training_loader.num_workers, cores)
        self.assertEqual(test_loader.num_workers, cores)

