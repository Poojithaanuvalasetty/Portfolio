CamVid Dataset Semantic Segmentation Project

Project Overview

This project focuses on semantic segmentation of urban scenes using the CamVid dataset. Semantic segmentation is critical in fields like autonomous driving, where understanding and categorizing different elements of an environment is essential. We utilized a modified FCN-ResNet50 model to segment the CamVid dataset into its labeled classes. The model was trained on Hopper Cluster, utilizing GPU acceleration for efficient training.

Dataset

About the Dataset

The Cambridge-driving Labeled Video Database (CamVid) provides labeled video sequences for urban scenes, where each pixel is annotated with one of 32 semantic classes. This dataset is often used in real-time semantic segmentation research.

Dataset Splits

	• Training: 367 images
	• Validation: 101 images
	• Test: 233 images

Citation

The CamVid dataset is provided by The University of Cambridge. For more details, please refer to the official dataset source:
CamVid Dataset

Running the Assignment on Hopper

Jupyter Notebook Setup

	1. File Upload: All required files have been uploaded to Hopper Cluster.
	2. Creating a GPU-Enabled Job: Start a GPU-supported Jupyter Notebook on Hopper.
	3. Pre-installed Libraries: No additional installations are necessary, as Hopper includes essential libraries such as PyTorch, NumPy, etc.

First Run Instructions

1. Install the Kaggle API to download the CamVid dataset. Uncomment the below set of lines in CamVid_Semantic_Segmentation.ipynb
!pip install kagglehub

# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("carlolepelaars/camvid")

# print("Path to dataset files:", path)

# import shutil
# import os

# # Path to the already downloaded dataset
# source_directory = path

# # Destination directory (current working directory './')
# destination_directory = './'

# # Check if the destination directory exists, if not, create it
# if not os.path.exists(destination_directory):
#     os.makedirs(destination_directory)

# # Move the dataset to the specified directory
# for filename in os.listdir(source_directory):
#     source_path = os.path.join(source_directory, filename)
#     destination_path = os.path.join(destination_directory, filename)
#     shutil.move(source_path, destination_path)

# print(f"Dataset moved to: {os.path.abspath(destination_directory)}")

2. Required Libraries:
Install the required Python libraries:
!pip install seaborn  # Only additional library required on Hopper Cluster
(Hopper Cluster has PyTorch installed by default.)

3. CUDA Device:
The code is executed on the Hopper Cluster, which supports CUDA, allowing for GPU acceleration.

Required Libraries

Here is a list of required libraries:
import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm  # Progress bar
from sklearn.metrics import confusion_matrix
import seaborn as sns

Project Structure

The repository is organized as follows:

	• CamVid_Semantic_Segmentation.ipynb: The main notebook that contains the code for data loading, preprocessing, training, validation, and evaluation of the segmentation model.
	• CamVid/: Contains the dataset with the following subdirectories:
	• train/: Training images.
	• train_labels/: Ground truth labels for training images.
	• val/: Validation images.
	• val_labels/: Ground truth labels for validation images.
	• test/: Test images.
	• test_labels/: Ground truth labels for test images.

Code and Execution Flow
	1. Data Preparation and Loading:
		• The dataset is processed to convert RGB labels into class IDs and prepared for training using PyTorch’s DataLoader.
		• Normalization and tensor transformations are applied to each image.
	2. Model Training and Validation:
		• We trained the FCN-ResNet50 model, leveraging a pre-trained ResNet50 backbone for feature extraction.
		• The model was trained over 25 epochs with the Adam optimizer, learning rate scheduler, and mixed-precision training for efficiency.
		• Training and validation metrics, such as accuracy and loss, are logged for each epoch.
	3. Testing and Visualization:
		• We evaluated the model on the test set, generating metrics including pixel accuracy, Intersection over Union (IoU), and confusion matrix per class.
		• Visualization outputs include the original image, true mask, predicted mask, and the predicted mask overlaid with class labels, providing a qualitative assessment of model performance.

Results and Analysis

1. Training and Validation Loss over Epochs:

	• Shows the model’s loss for each epoch, indicating model learning progress.

2. Training and Validation Accuracy over Epochs:
	• Shows the model’s loss for each epoch, indicating model learning progress.

2. Training and Validation Accuracy over Epochs:

	• Demonstrates pixel accuracy trends over epochs for both training and validation sets.

3. IoU per Class:

	• IoU scores per class are computed for the test set, indicating segmentation accuracy for each individual class.

4. Confusion Matrix:

	• Provides detailed insights into model predictions and highlights areas where certain classes are frequently confused.

5. Visual Comparisons:

	• Displays side-by-side images of the original, ground truth mask, predicted mask, and annotated predictions, which allow for qualitative assessment of the model’s segmentation quality.

Conclusion

Through this project, we successfully implemented a robust segmentation model using the CamVid dataset. The FCN-ResNet50 model demonstrated effective learning of complex urban classes, achieving high accuracy and IoU scores on major classes while highlighting areas for improvement on minor classes. Future work could involve exploring deeper models or data augmentation techniques to further enhance segmentation precision, especially for smaller classes.

