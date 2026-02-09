
import time
import os
from Classification.ComplexCNN import train_model
from functions.saveAndLoadFilteredData import write_results_to_file, FilteredNPZDataset
from functions.loadDatasetFunctions import DVSGestureNPYDataset

"""
When training, choose the filtered dataset you want to use 

- No filtering -> train_dataset_raw, test_dataset_raw
- OMS -> train_dataset_OMS, test_dataset_OMS
- Adaptive Elbow -> train_dataset_AdaptiveElbow, test_dataset_AdaptiveElbow
- Goal Oriented -> train_dataset_GoalOrientedThresholding, test_dataset_GoalOrientedThresholding
- Mean-StdDev -> train_dataset_MeanStd, test_dataset_MeanStd
- Global Saliency Crop -> train_dataset_GlobalSaliencyBasedCropping, test_dataset_GlobalSaliencyBasedCropping
- Denoised -> train_dataset_Denoise, test_dataset_Denoise
- Random Crop -> train_dataset_RandomCrop, test_dataset_RandomCrop
"""

# Raw (no filtering)

# ---- DVSGesture Dataset -----
print("Loading downsampled DVSGesture dataset...")

# training_ROOT = "C:/Users/giuli/Desktop/Giulia/PER/Event-Based-Project/DVSGestureDownsampled/ibmGestureTrain"
# testing_ROOT = "C:/Users/giuli/Desktop/Giulia/PER/Event-Based-Project/DVSGestureDownsampled/ibmGestureTest"

training_ROOT = "C:/Users/giuli/Desktop/Giulia/PER/Event-Based-Project/Datasets/ibmGestureTrain"
testing_ROOT = "C:/Users/giuli/Desktop/Giulia/PER/Event-Based-Project/Datasets/ibmGestureTest"

training_users = sorted(os.listdir(training_ROOT))
test_users = sorted(os.listdir(testing_ROOT))


# Training with raw downsampled dataset 

train_dataset_raw = DVSGestureNPYDataset(training_ROOT, users=training_users)
test_dataset_raw = DVSGestureNPYDataset(testing_ROOT, users=test_users)
# time training is calculated inside the train_model function, to include also the evaluation time, doesn't include the time taken to load the dataset
acc_raw, time_training_raw = train_model(train_dataset_raw, test_dataset_raw)
print(f"\nTime taken for training the ComplexCNN model: {time_training_raw:.2f} seconds")
write_results_to_file("ComplexCNN - Raw", acc_raw, time_training_raw)


# OMS 

train_dataset_OMS = FilteredNPZDataset("FilteredDatasets/OMS/train")
test_dataset_OMS  = FilteredNPZDataset("FilteredDatasets/OMS/test")
acc_OMS, time_training_OMS = train_model(train_dataset_OMS, test_dataset_OMS)
print(f"\nTime taken for training the ComplexCNN model: {time_training_OMS:.2f} seconds")   
write_results_to_file("ComplexCNN - OMS", acc_OMS, time_training_OMS)

# Adaptive Elbow

train_dataset_AdaptiveElbow = FilteredNPZDataset("FilteredDatasets/AdaptiveElbow/train")
test_dataset_AdaptiveElbow  = FilteredNPZDataset("FilteredDatasets/AdaptiveElbow/test")
acc_AdaptiveElbow, time_training_AdaptiveElbow = train_model(train_dataset_AdaptiveElbow, test_dataset_AdaptiveElbow)
print(f"\nTime taken for training the ComplexCNN model: {time_training_AdaptiveElbow:.2f} seconds")
write_results_to_file("ComplexCNN - Adaptive Elbow", acc_AdaptiveElbow, time_training_AdaptiveElbow)

# Training + evaluation time: 265.09 secondss
# Best Test Accuracy: 45.45%


# Goal Oriented Thresholding

train_dataset_GoalOrientedThresholding = FilteredNPZDataset("FilteredDatasets/GoalOrientedThresholding/train")
test_dataset_GoalOrientedThresholding  = FilteredNPZDataset("FilteredDatasets/GoalOrientedThresholding/test")
acc_GoalOrientedThresholding, time_training_GoalOrientedThresholding = train_model(train_dataset_GoalOrientedThresholding, test_dataset_GoalOrientedThresholding)
print(f"\nTime taken for training the ComplexCNN model: {time_training_GoalOrientedThresholding:.2f} seconds")  
write_results_to_file("ComplexCNN - Goal Oriented Thresholding", acc_GoalOrientedThresholding, time_training_GoalOrientedThresholding)  

# Mean Standard Deviation

train_dataset_MeanStd = FilteredNPZDataset("FilteredDatasets/MeanStd/train")
test_dataset_MeanStd  = FilteredNPZDataset("FilteredDatasets/MeanStd/test")
acc_MeanStd, time_training_MeanStd = train_model(train_dataset_MeanStd, test_dataset_MeanStd)
print(f"\nTime taken for training the ComplexCNN model: {time_training_MeanStd:.2f} seconds")
write_results_to_file("ComplexCNN - Mean Standard Deviation", acc_MeanStd, time_training_MeanStd)

# Global Saliency Based Cropping

train_dataset_GlobalSaliencyBasedCropping = FilteredNPZDataset("FilteredDatasets/GlobalSaliencyBasedCropping/train")
test_dataset_GlobalSaliencyBasedCropping  = FilteredNPZDataset("FilteredDatasets/GlobalSaliencyBasedCropping/test")
acc_GlobalSaliencyBasedCropping, time_training_GlobalSaliencyBasedCropping = train_model(train_dataset_GlobalSaliencyBasedCropping, test_dataset_GlobalSaliencyBasedCropping)
print(f"\nTime taken for training the ComplexCNN model: {time_training_GlobalSaliencyBasedCropping:.2f} seconds")
write_results_to_file("ComplexCNN - Global Saliency Based Cropping", acc_GlobalSaliencyBasedCropping, time_training_GlobalSaliencyBasedCropping)


# Denoise

train_dataset_Denoise = FilteredNPZDataset("FilteredDatasets/Denoise/train")
test_dataset_Denoise  = FilteredNPZDataset("FilteredDatasets/Denoise/test")
acc_Denoise, time_training_Denoise = train_model(train_dataset_Denoise, test_dataset_Denoise)
print(f"\nTime taken for training the ComplexCNN model: {time_training_Denoise:.2f} seconds")
write_results_to_file("ComplexCNN - Denoise", acc_Denoise, time_training_Denoise)

# Random Crop

train_dataset_RandomCrop = FilteredNPZDataset("FilteredDatasets/RandomCrop/train")
test_dataset_RandomCrop  = FilteredNPZDataset("FilteredDatasets/RandomCrop/test")
acc_RandomCrop, time_training_RandomCrop = train_model(train_dataset_RandomCrop, test_dataset_RandomCrop)
print(f"\nTime taken for training the ComplexCNN model: {time_training_RandomCrop:.2f} seconds")
write_results_to_file("ComplexCNN - Random Crop", acc_RandomCrop, time_training_RandomCrop)

