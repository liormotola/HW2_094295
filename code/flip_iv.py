import os
import shutil
import random
import numpy as np
from sklearn.model_selection import KFold
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms.functional as TF
import torch


def flip_iv_to_vi():
    # Set the paths for the original dataset and the destination directories for the new datasets

    original_dataset_path = os.path.join("..", "data","copy_of_new_train")
    class_path = os.path.join(original_dataset_path, "iv")
    new_path = os.path.join("..", "data","copy_of_new_train","vi_from_iv")
    os.makedirs(new_path, exist_ok=True)
    images = os.listdir(class_path)
    images = [os.path.join(class_path, img) for img in images]

    for image_path in images:
        new_image_path = os.path.join(new_path, os.path.basename(image_path))
        os.makedirs(os.path.dirname(new_image_path), exist_ok=True)
        image = Image.open(image_path)

        augmented_image = transforms.RandomHorizontalFlip(1)(image)
        # Save the augmented image
        augmented_image.save(new_image_path)


def flip_vi_to_iv():
    # Set the paths for the original dataset and the destination directories for the new datasets

    original_dataset_path = os.path.join("..", "data", "copy_of_new_train")
    class_path = os.path.join(original_dataset_path, "vi")
    new_path = os.path.join("..", "data", "copy_of_new_train", "iv_from_vi")
    os.makedirs(new_path, exist_ok=True)
    images = os.listdir(class_path)
    images = [os.path.join(class_path, img) for img in images]

    for image_path in images:
        new_image_path = os.path.join(new_path, os.path.basename(image_path))
        os.makedirs(os.path.dirname(new_image_path), exist_ok=True)
        image = Image.open(image_path)

        augmented_image = transforms.RandomHorizontalFlip(1)(image)
        # Save the augmented image
        augmented_image.save(new_image_path)

def flip_v_horizontal():
    # Set the paths for the original dataset and the destination directories for the new datasets

    original_dataset_path = os.path.join("..", "data", "copy_of_new_train")
    class_path = os.path.join(original_dataset_path, "v")
    new_path = os.path.join("..", "data", "copy_of_new_train", "v_from_v")
    os.makedirs(new_path, exist_ok=True)
    images = os.listdir(class_path)
    images = [os.path.join(class_path, img) for img in images]

    for image_path in images:
        new_image_path = os.path.join(new_path, os.path.basename(image_path)).replace(".png","_new.png")
        os.makedirs(os.path.dirname(new_image_path), exist_ok=True)
        image = Image.open(image_path)

        augmented_image = transforms.RandomHorizontalFlip(1)(image)
        # Save the augmented image
        augmented_image.save(new_image_path)

if __name__ == '__main__':
    # flip_iv_to_vi()
    # flip_vi_to_iv()
    flip_v_horizontal()