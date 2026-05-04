import random
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import albumentations
import datasets
from datasets import Dataset
from PIL import Image
import multiprocessing

from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner
from flwr_datasets.visualization import plot_label_distributions

#Constants
SIZE_IMG = 299


class FedISIC2019_Dataset():
    fds = None
    labels = None
    labels_for_the_labels = ["mel", "mel-nev", "bcc", "ak", "bk", "df", "vl", "scc"]
    amt_labels = 8
    seed = None

    def __init__(self, seed: int):
        self.seed = seed

        #Seeding RNG
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        else:
            print("Dataset RNG not seeded")
            exit(1)

        #Loading Fed-ISIC2019 via flwr_datasets
        self.fds = FederatedDataset(
            dataset="flwrlabs/fed-isic2019", 
            partitioners={
                "train" : NaturalIdPartitioner(partition_by="center"),
                "test" : NaturalIdPartitioner(partition_by="center")
            }
        )

    def get_partition_label_count(self, partition: Dataset, partition_id: int, quiet_output = True):
        if(not quiet_output):
            print(f"Counting Labels for Partition {partition_id}")

        label_counters = [0 for n in range(self.amt_labels)]
        for row in partition:
            label_counters[row["label"]] += 1
        return label_counters
    
    def __to_numpy(self, img):
        if isinstance(img, np.ndarray):
            return img
        return np.array(img)

    def apply_train_val_test_standard_transform(self, pil_img):
        transform = albumentations.Compose([
            albumentations.PadIfNeeded(min_height=SIZE_IMG, min_width=SIZE_IMG, border_mode=0),
            albumentations.CenterCrop(height=SIZE_IMG, width=SIZE_IMG)
        ], seed=self.seed)

        #Taking the Pillow formated image from the dataset and make it into a Numpy Array
        img_np = self.__to_numpy(pil_img)
        
        #Applying the transform
        augmented = transform(image=img_np)["image"]

        #Transforming back into PIL Image for memory conservation
        augmented_pil_img = Image.fromarray(augmented)
        
        return augmented_pil_img
    
    def __map_image_to_standard_transformed_image(self, row):
        row["image"] = self.apply_train_val_test_standard_transform(row["image"])
        return row

    def augment_dataset(self, quiet_output = False):
        #Stage 1 - Loading a Partiton, Standardizing and counting labels.
        num_of_partitions = self.fds.partitioners["train"].num_partitions
        partition_label_counts = [[0 for n in range(self.amt_labels)] for n in range(num_of_partitions)]

        for partition_index in range(num_of_partitions):
            partition_data = self.fds.load_partition(partition_id=partition_index, split="train")
            partition_label_counts[partition_index] = self.get_partition_label_count(partition=partition_data, partition_id=partition_index, quiet_output=quiet_output)
            standardized_dataset = partition_data.map(self.__map_image_to_standard_transformed_image, num_proc=4)
            standardized_dataset.save_to_disk(f"datset_proccesed_data/partition{partition_index}")
        
        return


if __name__ == "__main__":
    dataset = FedISIC2019_Dataset(67)
    dataset.augment_dataset()