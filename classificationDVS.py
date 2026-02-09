
import time
from functions.loadDatasetFunctions import FilteredNPZDataset
from Classification.ComplexCNN import train_model
from functions.saveAndLoadFilteredData import write_results_to_file

"""
When training, choose the filtered dataset you want to use 

- No filtering -> train_dataset_raw, test_dataset_raw
- OMS -> filtered_events_OMS_train, filtered_events_OMS_test
- Adaptive Elbow -> filtered_events_adaptiveElbow_train, filtered_events_adaptiveElbow_test
- Goal Oriented -> filtered_events_GoalOriented_train, filtered_events_Goal
- Mean-StdDev -> filtered_events_MeanStd_train, filtered_events_MeanStd_test
- Global Saliency Crop -> filtered_events_GlobalSaliencyCrop_train, filtered_events_GlobalSaliencyCrop_test
- Denoised -> filtered_events_Denoised_train, filtered_events_Denoised_test
- Random Crop -> filtered_events_RandomCrop_train, filtered_events_RandomCrop_test
"""

# Raw (no filtering)

# OMS 

train_dataset_OMS = FilteredNPZDataset("FilteredDatasets/OMS/train")
test_dataset_OMS  = FilteredNPZDataset("FilteredDatasets/OMS/test")
start_time_training = time.time()
history_OMS = train_model(train_dataset_OMS, test_dataset_OMS)
end_time_training = time.time()
time_training_OMS = end_time_training - start_time_training
print(f"\nTime taken for training the ComplexCNN model: {time_training_OMS:.2f} seconds")   
write_results_to_file("ComplexCNN - OMS", history_OMS, time_training_OMS)

# Adaptive Elbow

train_dataset_AdaptiveElbow = FilteredNPZDataset("FilteredDatasets/AdaptiveElbow/train")
test_dataset_AdaptiveElbow  = FilteredNPZDataset("FilteredDatasets/AdaptiveElbow/test")
start_time_training = time.time()
history_AdaptiveElbow = train_model(train_dataset_AdaptiveElbow, test_dataset_AdaptiveElbow)
end_time_training = time.time()
time_training_AdaptiveElbow = end_time_training - start_time_training
print(f"\nTime taken for training the ComplexCNN model: {time_training_AdaptiveElbow:.2f} seconds")
write_results_to_file("ComplexCNN - Adaptive Elbow", history_AdaptiveElbow, time_training_AdaptiveElbow)

# Training + evaluation time: 265.09 secondss
# Best Test Accuracy: 45.45%


# Goal Oriented Thresholding

train_dataset_GoalOrientedThresholding = FilteredNPZDataset("FilteredDatasets/GoalOrientedThresholding/train")
test_dataset_GoalOrientedThresholding  = FilteredNPZDataset("FilteredDatasets/GoalOrientedThresholding/test")

start_time_training = time.time()
history_GoalOrientedThresholding = train_model(train_dataset_GoalOrientedThresholding, test_dataset_GoalOrientedThresholding)
end_time_training = time.time()
time_training_GoalOrientedThresholding = end_time_training - start_time_training
print(f"\nTime taken for training the ComplexCNN model: {time_training_GoalOrientedThresholding:.2f} seconds")  
write_results_to_file("ComplexCNN - Goal Oriented Thresholding", history_GoalOrientedThresholding, time_training_GoalOrientedThresholding)  

# Mean Standard Deviation

train_dataset_MeanStd = FilteredNPZDataset("FilteredDatasets/MeanStd/train")
test_dataset_MeanStd  = FilteredNPZDataset("FilteredDatasets/MeanStd/test")
start_time_training = time.time()
history_MeanStd = train_model(train_dataset_MeanStd, test_dataset_MeanStd)
end_time_training = time.time()
time_training_MeanStd = end_time_training - start_time_training
print(f"\nTime taken for training the ComplexCNN model: {time_training_MeanStd:.2f} seconds")
write_results_to_file("ComplexCNN - Mean Standard Deviation", history_MeanStd, time_training_MeanStd)

# Global Saliency Based Cropping

train_dataset_GlobalSaliencyBasedCropping = FilteredNPZDataset("FilteredDatasets/GlobalSaliencyBasedCropping/train")
test_dataset_GlobalSaliencyBasedCropping  = FilteredNPZDataset("FilteredDatasets/GlobalSaliencyBasedCropping/test")
start_time_training = time.time()
history_GlobalSaliencyBasedCropping = train_model(train_dataset_GlobalSaliencyBasedCropping, test_dataset_GlobalSaliencyBasedCropping)
end_time_training = time.time()
time_training_GlobalSaliencyBasedCropping = end_time_training - start_time_training
print(f"\nTime taken for training the ComplexCNN model: {time_training_GlobalSaliencyBasedCropping:.2f} seconds")
write_results_to_file("ComplexCNN - Global Saliency Based Cropping", history_GlobalSaliencyBasedCropping, time_training_GlobalSaliencyBasedCropping)


# Denoise

train_dataset_Denoise = FilteredNPZDataset("FilteredDatasets/Denoise/train")
test_dataset_Denoise  = FilteredNPZDataset("FilteredDatasets/Denoise/test")
start_time_training = time.time()
history_Denoise = train_model(train_dataset_Denoise, test_dataset_Denoise)
end_time_training = time.time()
time_training_Denoise = end_time_training - start_time_training
print(f"\nTime taken for training the ComplexCNN model: {time_training_Denoise:.2f} seconds")
write_results_to_file("ComplexCNN - Denoise", history_Denoise, time_training_Denoise)

# Random Crop

train_dataset_RandomCrop = FilteredNPZDataset("FilteredDatasets/RandomCrop/train")
test_dataset_RandomCrop  = FilteredNPZDataset("FilteredDatasets/RandomCrop/test")
start_time_training = time.time()
history_RandomCrop = train_model(train_dataset_RandomCrop, test_dataset_RandomCrop)
end_time_training = time.time()
time_training_RandomCrop = end_time_training - start_time_training
print(f"\nTime taken for training the ComplexCNN model: {time_training_RandomCrop:.2f} seconds")
write_results_to_file("ComplexCNN - Random Crop", history_RandomCrop, time_training_RandomCrop)







# Random Crop 
# train_dataset = FilteredNPZDataset("FilteredDatasets/RandomCrop/train")
# test_dataset  = FilteredNPZDataset("FilteredDatasets/RandomCrop/test")

