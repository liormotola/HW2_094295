import copy
import os
import time
import os
import random
import shutil
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from tqdm import tqdm

np.random.seed(0)
torch.manual_seed(0)

base_dir = os.path.join("..", "data")


def split_data(source_dir, train_dir, test_dir, split_ratio):
    # Create train and test directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Iterate over the source directory and its subdirectories
    for root, dirs, files in os.walk(source_dir):
        # Exclude the root/source directory itself
        if root != source_dir:
            # Create corresponding subdirectories in train and test directories
            train_subdir = os.path.join(train_dir, os.path.relpath(root, source_dir))
            test_subdir = os.path.join(test_dir, os.path.relpath(root, source_dir))
            os.makedirs(train_subdir, exist_ok=True)
            os.makedirs(test_subdir, exist_ok=True)

            # Randomly shuffle the list of files in the current subdirectory
            random.shuffle(files)

            # Calculate the number of files for the train and test sets based on the split ratio
            num_train_files = int(len(files) * split_ratio)
            train_files = files[:num_train_files]
            test_files = files[num_train_files:]

            # Move files to the train directory
            for file in train_files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(train_subdir, file)
                shutil.copy(src_file, dst_file)

            # Move files to the test directory
            for file in test_files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(test_subdir, file)
                shutil.copy(src_file, dst_file)

# Set the paths and split ratio
source_dir = os.path.join("..", "data","train")
train_dir = os.path.join("..", "data","new_train")
test_dir = os.path.join("..", "data","test")
split_ratio = 0.8  # 80% for train, 20% for test

# Call the function to split the data
split_data(source_dir, train_dir, test_dir, split_ratio)