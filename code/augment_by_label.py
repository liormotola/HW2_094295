import os
import shutil
import numpy as np
from sklearn.model_selection import KFold
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms.functional as TF
import torch


def create_datasets(original_dataset_path,
                    dataset_names,
                    augmentation_transforms,
                    flip_transforms,
                    together_transforms,
                    together_class,
                    flip_v):
    """
     split original data into 3 folds, create 3 new datasets where each has 2 train folds and one val fold
     and perform augmentations on train data only for each dataset.
    The new datasets will be saved in dataset_names
    This function holds specific augmentations for each class!
    @param: original_dataset_path - path to the original data directory
    @param: dataset_names - list of 3 names, one for each dataset created
    @param: augmentation_transforms: list of augmentations to perform over train data, will be performed over all labels,
     each augmentation separately.
    @param: flip_transforms: list of flip transformations to perform over train data. will be performed only over the labels ["i","ii","iii","x"]
    @param: together_transforms: list of augmentations to perform over train data.
            Augmentations will be performed one over each other (one augmented image wil be created). will be performed over labels in @together_classes only
    @param: together_classes: list of labels on which together_transforms will be performed.
    @flip_v : boolean argument indicating whether to perform horizontal flip over the labels [v,iv,vi] or not.
    """

    new_dataset_paths = [os.path.join("..", "data", name) for name in dataset_names]

    # Set the number of folds for k-fold cross-validation
    k = 3

    flip_class = ["i", "x"]
    classes = os.listdir(original_dataset_path)
    original_images = []
    for class_name in classes:
        class_path = os.path.join(original_dataset_path, class_name)
        images = os.listdir(class_path)
        images = [os.path.join(class_path, img) for img in images]
        original_images.extend(images)

    # Create the k-fold cross-validation indices
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    indices = np.arange(len(original_images))

    # Iterate over the k-fold indices and create new datasets
    for i, (train_indices, val_indices) in enumerate(kf.split(indices)):

        new_dataset_path = new_dataset_paths[i]
        os.makedirs(new_dataset_path, exist_ok=True)

        train_dir = os.path.join(new_dataset_path, "train")
        val_dir = os.path.join(new_dataset_path, "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        # Copy and augment the corresponding image files to the train directory
        for idx in train_indices:
            image_path = original_images[idx]
            class_name = os.path.basename(os.path.dirname(image_path))
            new_image_path = os.path.join(train_dir, class_name, os.path.basename(image_path))
            os.makedirs(os.path.dirname(new_image_path), exist_ok=True)

            shutil.copy(image_path, new_image_path)

            # augmentations specific for class "ii"
            if class_name == "ii":
                ii_transforms = [add_noise, exposure, transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 5)),
                                 transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 5))] + flip_transforms

                for n, transformation in enumerate(ii_transforms):
                    image = Image.open(image_path)
                    augmented_image = transformation(image)
                    augmented_image.save(new_image_path.replace(".png", f"_{n}.png"))

            # augmentations specific for class "iii"
            elif class_name == "iii":
                iii_transforms = [add_noise, transforms.RandomRotation(degrees=5),
                                  transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 5))] + flip_transforms

                for n, transformation in enumerate(iii_transforms):
                    image = Image.open(image_path)
                    augmented_image = transformation(image)
                    augmented_image.save(new_image_path.replace(".png", f"_{n}.png"))

            # all other classes
            else:
                for n, transformation in enumerate(augmentation_transforms):
                    image = Image.open(image_path)
                    augmented_image = transformation(image)
                    augmented_image.save(new_image_path.replace(".png", f"_{n}.png"))

                if class_name in flip_class:
                    for n, transformation in enumerate(flip_transforms):
                        image = Image.open(image_path)
                        augmented_image = transformation(image)
                        augmented_image.save(new_image_path.replace(".png", f"_{n + len(augmentation_transforms)}.png"))

                if class_name == "i" or class_name == "ix":
                    aug = transforms.Compose(
                        [transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 5)), transforms.RandomRotation(degrees=30)])
                    image = Image.open(image_path)
                    augmented_image = aug(image)
                    augmented_image.save(new_image_path.replace(".png", f"_aug.png"))
                else:
                    aug = transforms.Compose(
                        [transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 5)), transforms.RandomRotation(degrees=15)])
                    image = Image.open(image_path)
                    augmented_image = aug(image)
                    augmented_image.save(new_image_path.replace(".png", f"_aug.png"))

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

            if together_class and class_name in together_class:
                image = Image.open(image_path)
                augmented_image = together_transforms[0](image)
                for transformation in together_transforms[1:]:
                    augmented_image = transformation(augmented_image)
                augmented_image.save(
                    new_image_path.replace(".png", f"_{1 + len(augmentation_transforms) + len(flip_transforms)}.png"))

        # validation directory
        for idx in val_indices:
            image_path = original_images[idx]
            class_name = os.path.basename(os.path.dirname(image_path))
            new_image_path = os.path.join(val_dir, class_name, os.path.basename(image_path))
            os.makedirs(os.path.dirname(new_image_path), exist_ok=True)

            shutil.copy(image_path, new_image_path)


def exposure(image):
    """
    perform exposure augmentation over an image. intensity = 2
    """

    image_tensor = TF.to_tensor(image)
    exposure_factor = 2
    adjusted_image_tensor = image_tensor * exposure_factor
    # Clip the pixel values to the valid range [0, 1]
    adjusted_image_tensor = torch.clamp(adjusted_image_tensor, 0.0, 1.0)
    adjusted_image = TF.to_pil_image(adjusted_image_tensor)

    return adjusted_image


def add_noise(image):
    """
    add noise to a given image. intensity = 0.1
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


if __name__ == '__main__':
    original_dataset_path = os.path.join("..", "data", "new_train")

    flip_transform = [
        transforms.RandomHorizontalFlip(1),
        transforms.RandomVerticalFlip(1),
    ]

    # group 5
    names5 = ["augments_group_5_1", "augments_group_5_2", "augments_group_5_3"]

    together_transform = [transforms.RandomHorizontalFlip(1), add_noise]
    create_datasets(original_dataset_path=original_dataset_path, dataset_names=names5,
                    augmentation_transforms=[transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 5)), add_noise,
                                             transforms.RandomRotation(degrees=15)],
                    flip_transforms=flip_transform,
                    together_transforms=together_transform,
                    together_class=["i"], flip_v=True)
