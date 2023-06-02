import os
import shutil
import random
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms.functional as TF
import torch

np.random.seed(0)
random.seed(0)

def create_augmented_dataset(original_dataset_path,
                             augmentation_transforms,
                             flip_transforms,
                             flip_v):
    """
    Create augmented training dataset from data in original_dataset_path.
    The new dataset will be created in a directory called "final_augmented_data/train"

    @param: original_dataset_path - path to the original data directory
    @param: augmentation_transforms: list of augmentations to perform, will be performed over all labels,
     each augmentation separately.
    @param: flip_transforms: list of flip transformations to perform. will be performed only over the labels ["i","ii","iii","x"]
    @flip_v : boolean argument indicating whether to perform horizontal flip over the labels [v,iv,vi] or not.
    """

    new_dataset_path = os.path.join("..", "data", "final_augmented_data")

    flip_class = ["i", "ii", "iii", "x"]
    classes = os.listdir(original_dataset_path)
    original_images = []
    for class_name in classes:
        class_path = os.path.join(original_dataset_path, class_name)
        images = os.listdir(class_path)
        images = [os.path.join(class_path, img) for img in images]
        original_images.extend(images)

    os.makedirs(new_dataset_path, exist_ok=True)
    train_dir = os.path.join(new_dataset_path, "train")
    os.makedirs(train_dir, exist_ok=True)

    # Copy and augment the corresponding image files to the train directory
    for image_path in original_images:
        class_name = os.path.basename(os.path.dirname(image_path))
        new_image_path = os.path.join(train_dir, class_name, os.path.basename(image_path))
        os.makedirs(os.path.dirname(new_image_path), exist_ok=True)

        shutil.copy(image_path, new_image_path)
        for n, transformation in enumerate(augmentation_transforms):
            # Apply augmentation transforms to the image
            image = Image.open(image_path)
            augmented_image = transformation(image)
            # Save the augmented image
            augmented_image.save(new_image_path.replace(".png", f"_{n}.png"))

        if class_name in flip_class:
            for n, transformation in enumerate(flip_transforms):
                image = Image.open(image_path)
                augmented_image = transformation(image)
                augmented_image.save(new_image_path.replace(".png", f"_{n + len(augmentation_transforms)}.png"))

        if flip_v:
            if class_name == "v":
                image = Image.open(image_path)
                augmented_image = transforms.RandomHorizontalFlip(1)(image)
                augmented_image.save(new_image_path.replace(".png", f"_new.png"))
            elif class_name == "iv":
                image = Image.open(image_path)
                augmented_image = transforms.RandomHorizontalFlip(1)(image)
                path = os.path.join(train_dir, "vi", os.path.basename(image_path))
                os.makedirs(os.path.dirname(path), exist_ok=True)
                augmented_image.save(path.replace(".png", f"_new.png"))
            elif class_name == "vi":
                image = Image.open(image_path)
                augmented_image = transforms.RandomHorizontalFlip(1)(image)
                path = os.path.join(train_dir, "iv", os.path.basename(image_path))
                os.makedirs(os.path.dirname(path), exist_ok=True)
                augmented_image.save(path.replace(".png", f"_new.png"))


def add_noise(image):
    """
    add noise to a given image
    """
    image_tensor = TF.to_tensor(image)
    # Generate random Gaussian noise
    noise_tensor = torch.randn_like(image_tensor)

    noise_intensity = 0.1
    noisy_image_tensor = image_tensor + noise_intensity * noise_tensor

    # Clip the pixel values to the valid range [0, 1]
    noisy_image_tensor = torch.clamp(noisy_image_tensor, 0.0, 1.0)
    noisy_image = TF.to_pil_image(noisy_image_tensor)
    return noisy_image

def reduce_i(dir):
    """
    delete 100 images from class i in train data after augmentations, to meet size limitations
    """
    class_path = os.path.join(dir,"train", "i")
    images = os.listdir(class_path)
    images = [os.path.join(class_path, img) for img in images]
    delete_images = random.sample(images, k=100)
    for image in delete_images:
        os.remove(image)


if __name__ == '__main__':
    original_dataset_path = os.path.join("..", "data", "new_train")

    flip_transform = [
        transforms.RandomHorizontalFlip(1),
        transforms.RandomVerticalFlip(1),
    ]
    combined_augmentation = transforms.Compose(
        [transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 5)), transforms.RandomRotation(degrees=15)])

    create_augmented_dataset(original_dataset_path=original_dataset_path,
                             augmentation_transforms=[combined_augmentation,
                                                      transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 5)), add_noise,
                                                      transforms.RandomRotation(degrees=15)],
                             flip_transforms=flip_transform,
                             flip_v=True)

    new_dir = os.path.join("..", "data", "final_augmented_data")
    reduce_i(new_dir)
