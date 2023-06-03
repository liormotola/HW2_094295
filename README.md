# HW2_094295
## Goal
Improve performance of ResNet50 classification over Roman-Figures-images dataset, by only applying changes to the dataset itself.

## Description
We were given a dataset and a code to train and eval the model over the dataset.
After performing initial cleaning and relabeling to the dataset manually we performed different augmentations over the dataset aiming to improve model's performances.
This Repository contains all code we used to perform our experiments.
<br>
#### This Reopsitory contains the following: 
  * hw2_environment.yml - contains all modules required to run this project.

  #### Code directory
    * `initial_train_test_split.py` - code to split the data into train and test set, and save it in the directories structure needed for the model's training.
    * `create_new_datasets.py` & `augment_by_label.py` - code to create the augmented datasets that served us in experiments 1-4 & 5 correspondingly. Saves 3 versions for each trial for CV process. 
     New data is saved in the required directories structure.
    * `evaluate.py` - Code to evaluate a pretrained model. Generates confusion matrix and prints accuracy per label.
    * `create_final_train_data.py` - generates our final train and validation datasets in the required structure.
    * `our_run_train_eval.py` - same code we were given to train and eval the model with minor changes. Number of epoches was reduced to 20. Expects to get 4 argumnets:<br>
        `python our_run_train_eval.py <path to data directory> <location to save the trained model> <location to save loss graph> <location to save accuracy graph>`

