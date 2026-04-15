import unittest
from dataset.dataset import FedISIC2019_Dataset

class TestDataset(unittest.TestCase):
    CENTRALIZED_TRAIN_IMG_COUNT = 18597
    CENTRALIZED_TEST_IMG_COUNT = 4650

    def test_centralized_dataset(self):
        #Instaiate a dataset object
        dataset = FedISIC2019_Dataset()
        
        #Call to fetch the centralized verison of the dataset
        full_train, full_test = dataset.centralized_dataset()

        train_img_counter = 0
        test_img_counter = 0
        #Assert
        for img in full_train:
            train_img_counter += 1

        for img in full_test:
            test_img_counter += 1
        
        self.assertEqual(train_img_counter, self.CENTRALIZED_TRAIN_IMG_COUNT)
        self.assertEqual(test_img_counter, self.CENTRALIZED_TEST_IMG_COUNT)
        