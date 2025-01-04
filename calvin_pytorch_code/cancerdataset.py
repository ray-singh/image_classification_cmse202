# defines CancerDataset class for loading in data that works with pytorch
# methods. 

import torch
import os
import pandas as pd
from torchvision.io import read_image
import os
import glob
import numpy as np
from torch.utils.data import Dataset

#file_nums = "/Users/calvindejong/Downloads/cancer_images"

def get_cancer_image_paths(path_to_IDC_regular):
    # returns tuple (train_image_paths, test_image_paths) given a path to the
    # cancer data (https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)
    file_nums = os.listdir(path_to_IDC_regular)
    return (file_nums[::2], file_nums[1::2])
    
    



class CancerDataset(Dataset):
    def __init__(self, img_labels, img_paths, transform=None, target_transform=None):
        # img-labels
        self.img_labels = img_labels
        self.img_paths = img_paths
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = read_image(img_path)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

