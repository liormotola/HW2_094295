# HW2_094295
## Goal
Improve classification performance of a ResNet50 model over a Roman-Figures-images dataset, by applying changes solely to the dataset itself.

## Description
We were given a dataset and a code to train and evaluate the model over the dataset.
After performing an initial cleaning and relabeling to the dataset manually, we performed different augmentations over the dataset aiming to improve the model's performance.<br>
This Repository contains all code we used to perform our experiments.
<br>
#### The Repository contains the following: 
  * `hw2_environment.yml` - contains all modules required to run this project.

  #### Code directory

  * All code files:
  <ul>


  * `initial_train_test_split.py` - code to split the data into train and test sets, and save it in the directories' structure needed for the model's training.<br>
  * `create_new_datasets.py` & `augment_by_label.py` - code to create the augmented datasets that served us in experiments 1-4 & 5 respectively. Saves 3 versions for each experiment for the CV process. New data is being saved in the required directories' structure.<br>
  * `evaluate.py` - Code to evaluate a pre-trained model. Generates confusion matrix and prints accuracy per label.
  * `create_final_train_data.py` - generates our final train and validation datasets in the required structure.
  * `our_run_train_eval.py` - the same code we were given to train and evaluate the model with minor changes. The number of epochs was reduced to 20. Expects to get 4 arguments:<br>
        `python our_run_train_eval.py <path to data directory> <location to save the trained model> <location to save loss graph> <location to save accuracy graph>`
  * `run_cv.sh` - Code used to run the cv process.

</ul>
 
* Important Note: all code files should be run from inside the code directory, and the data directory should be outside of the code directory.
