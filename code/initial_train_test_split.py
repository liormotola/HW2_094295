import os
import random
import shutil
import numpy as np
import torch


np.random.seed(0)
torch.manual_seed(0)

base_dir = os.path.join("..", "data")


def split_data(source_dir, train_dir, test_dir, split_ratio):
    """
    split data into train and test datasets and saves it to new directories.
    @param: source_dir = path to original data
    @param: train_dir = path to save new train data
    @param: test_dir = path to save new test data
    @param: split_ratio = split_ratio of the original dataset wil be saved as training data, the rest will be used as test data.
    """

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Iterate over the source directory and its subdirectories
    for root, dirs, files in os.walk(source_dir):
        if root != source_dir:
            train_subdir = os.path.join(train_dir, os.path.relpath(root, source_dir))
            test_subdir = os.path.join(test_dir, os.path.relpath(root, source_dir))
            os.makedirs(train_subdir, exist_ok=True)
            os.makedirs(test_subdir, exist_ok=True)

            # Randomly shuffle the list of files in the current subdirectory
            random.shuffle(files)

            num_train_files = int(len(files) * split_ratio)
            train_files = files[:num_train_files]
            test_files = files[num_train_files:]

            for file in train_files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(train_subdir, file)
                shutil.copy(src_file, dst_file)

            for file in test_files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(test_subdir, file)
                shutil.copy(src_file, dst_file)

if __name__ == '__main__':

    source_dir = os.path.join("..", "data","train")
    train_dir = os.path.join("..", "data","new_train")
    test_dir = os.path.join("..", "data","test")
    split_ratio = 0.8  # 80% for train, 20% for test

    split_data(source_dir, train_dir, test_dir, split_ratio)