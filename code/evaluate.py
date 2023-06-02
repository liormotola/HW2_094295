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
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(0)
torch.manual_seed(0)
BATCH_SIZE = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_datasets(data_dir):
    """Loads and transforms the datasets."""
    # Resize the samples and transform them into tensors
    data_transforms = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()])

    # Create a pytorch dataset from a directory of images
    dataset = datasets.ImageFolder(data_dir, data_transforms)

    return dataset

def load_model(saved_model, num_classes):
    # Use a prebuilt pytorch's ResNet50 model
    model_ft = models.resnet50(pretrained=False)

    # Fit the last layer for our specific task
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    model_ft.load_state_dict(torch.load(saved_model))
    model_ft = model_ft.to(device)

    return model_ft

def plot_confusion_mat(true_labels, predicted_labels,title,set_type):
    conf_mat = confusion_matrix(true_labels, predicted_labels)
    ax = sns.heatmap(conf_mat, annot=True, fmt='d')
    plt.ylabel("True Values")
    plt.xlabel("Predicted Values")
    ax.set_xticklabels( ['i', 'ii', 'iii', 'iv', 'ix', 'v', 'vi', 'vii', 'viii', 'x'])
    ax.set_yticklabels(['i', 'ii', 'iii', 'iv', 'ix', 'v', 'vi', 'vii', 'viii', 'x'])
    plt.title(f"{title}-{set_type}")
    plt.savefig(f"{title} {set_type}-confusion matrix")
    plt.show()

def set_eval(data_dir, saved_model,title,set_type):

    val_dataset = load_datasets(data_dir)
    print(f"test size: {len(val_dataset)}")
    class_names = val_dataset.classes
    print("The classes are: ", class_names)
    num_classes = len(class_names)
    model_ft = load_model(saved_model,num_classes=num_classes)
    model_ft.eval()
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    num_correct = 0
    num_total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model_ft(images)
            _, predicted = torch.max(outputs, 1)

            num_correct += (predicted == labels).sum().item()
            num_total += labels.size(0)
            predicted_labels.extend(predicted)
            true_labels.extend(labels)

            for i in range(labels.size(0)):
                label = labels[i]
                prediction = predicted[i]
                class_correct[label] += (prediction == label).item()
                class_total[label] += 1

    accuracy = num_correct / num_total
    print('Overall Accuracy: {:.2%}'.format(accuracy))

    for i in range(num_classes):
        class_accuracy = class_correct[i] / class_total[i]
        print('Class {} Accuracy: {:.2%}'.format(i, class_accuracy))

    plot_confusion_mat(true_labels=true_labels,
                       predicted_labels=predicted_labels,
                       title=title,
                       set_type=set_type)

if __name__ == '__main__':

    data_dirs = ["augments_group_5_1", "augments_group_5_2", "augments_group_5_3"]

    for dir in data_dirs:
        data_dir = os.path.join("..", "data",dir,"val")
        saved_model = os.path.join("..", "models",f"{dir}.pt")
        set_eval(data_dir=data_dir,
                 saved_model=saved_model,
                 title=dir,set_type="validation")

        test_dir =  os.path.join("..", "data","test")
        set_eval(data_dir=test_dir,
                 saved_model=saved_model,
                 title=dir, set_type="test")

