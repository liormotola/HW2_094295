from sklearn.model_selection import KFold
import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from tqdm import tqdm
import os
import shutil
import random
import numpy as np
from sklearn.model_selection import KFold
import torchvision.transforms as transforms
from PIL import Image


np.random.seed(0)
torch.manual_seed(0)

print("Your working directory is: ", os.getcwd())

# Training hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 20
LR = 0.001
num_folds = 3

base_dir = os.path.join("..", "data")
train_dir = os.path.join(base_dir, "new_train")
test_dir = os.path.join(base_dir, "test")

kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

def load_datasets(train_dir, val_dir):
    """Loads and transforms the datasets."""
    # Resize the samples and transform them into tensors
    data_transforms = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()])

    # Create a pytorch dataset from a directory of images
    train_dataset = datasets.ImageFolder(train_dir, data_transforms)
    val_dataset = datasets.ImageFolder(val_dir, data_transforms)

    return train_dataset, val_dataset

def create_dataset(root_dir):
    os.mkdir()