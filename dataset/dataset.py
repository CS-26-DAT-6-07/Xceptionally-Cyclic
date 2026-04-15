import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import albumentations
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner

#Constants
SIZE_IMG = 299


class FedISIC2019_Dataset():
    fds = None
    labels = None

    def __init__(self):
        #Loading Fed-ISIC2019 via flwr_datasets
        self.fds = FederatedDataset(
            dataset="flwrlabs/fed-isic2019", 
            partitioners={
                "train" : NaturalIdPartitioner(partition_by="center"),
                "test" : NaturalIdPartitioner(partition_by="center")
            }
        )

    def centralized_dataset(self):
        full_train = self.fds.load_split("train")
        full_test = self.fds.load_split("test")

        self.labels = full_train.features["label"].names
        return full_train, full_test
    
    def plot_centralized_train_class_distribution(self):
        full_train = self.centralized_dataset()
        label_counter = [0, 0, 0, 0, 0, 0, 0, 0]
        
        for sample in full_train:
            label_on_sample = sample["label"]
            label_counter[label_on_sample] += 1
        
        





full_train, full_test = FedISIC2019_Dataset().centralized_dataset()

print(full_train[0]["label"])