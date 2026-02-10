import os
import time
import matplotlib as plt

from functions.saveAndLoadFilteredData import FilteredNPZDataset, save_filtered_dataset
from functions.loadDatasetFunctions import DVSGestureNPYDataset
from Classification.ComplexCNN import train_model
from Filtering_techniques.MaskGoalOriented import MaskGoalOrientedOMSFiltering
from Filtering_techniques.MaskMeanStandardDeviation import MaskMeanStandardDeviation
from Filtering_techniques.MaskGlobalSaliencyBasedCropping import MaskGlobalSaliencyBasedCropping
from functions.writeResultsFunctions import write_parameter_tuning_results_to_file

# --- Goal Oriented Thresholding ---

def GoalOriented_filtering_pipeline(train_dataset_raw, test_dataset_raw, keep_percent, filtering_root = "FilterTuning", scale_factor=3):

    filtered_events_GoalOriented_train = []
    filtered_events_GoalOriented_test = []
    Err_list_GoalOriented = []
    
    start_time_GoalOriented = time.time()
    for event, label in train_dataset_raw:
        # Initialize and run Goal Oriented Thresholding
        GoalOrientedFilter = MaskGoalOrientedOMSFiltering(event, scale_factor)
        filtered_events, masked_OMS, OMSMap, ERR_goal = GoalOrientedFilter.Goadaptive_thresholding(keep_percent) 
        # GoalOrientedFilter.GoalOriented_filtering_visualization(OMSMap, masked_OMS)
        filtered_events_GoalOriented_train.append((filtered_events, label))
        Err_list_GoalOriented.append(ERR_goal)
    for event, label in test_dataset_raw:
        # Initialize and run Goal Oriented Thresholding
        GoalOrientedFilter = MaskGoalOrientedOMSFiltering(event, scale_factor)
        filtered_events, masked_OMS, OMSMap, ERR_goal = GoalOrientedFilter.Goadaptive_thresholding(keep_percent) 
        # GoalOrientedFilter.GoalOriented_filtering_visualization(OMSMap, masked_OMS)
        filtered_events_GoalOriented_test.append((filtered_events, label))
        Err_list_GoalOriented.append(ERR_goal)
    end_time_GoalOriented = time.time()
    time_GoalOriented = end_time_GoalOriented - start_time_GoalOriented

    average_ERR_GoalOriented = sum(Err_list_GoalOriented) / len(Err_list_GoalOriented)
    print(f"\nAverage Goal-Oriented Filtering Error (ERR) across all events: {average_ERR_GoalOriented:.4f}")
    print(f"time Goal Oriented Filtering = {time_GoalOriented:.2f} seconds")

    save_filtered_dataset(
        filtered_events_GoalOriented_train,
        save_dir=f"{filtering_root}/Threshold_{keep_percent}/train",
        prefix="train"
    )

    save_filtered_dataset(
        filtered_events_GoalOriented_test,
        save_dir=f"{filtering_root}/Threshold_{keep_percent}/test",
        prefix="test"
    )

    return average_ERR_GoalOriented, time_GoalOriented


def MeanStd_filtering_pipeline(train_dataset_raw, test_dataset_raw, k_sigma, filtering_root = "FilterTuning", scale_factor=3):
    
    filtered_events_MeanStd_train = []
    filtered_events_MeanStd_test = []
    Err_list_MeanStd = []
    
    start_time_MeanStd = time.time()
    # Initialize and run Mean and Standard Deviation Thresholding
    for event, label in train_dataset_raw:
        MeanStdFilter = MaskMeanStandardDeviation(event, scale_factor)
        filtered_events, ERR_MStd = MeanStdFilter.Mean_std_thresholding(k_sigma)
        # MeanStdFilter.MeanStd_filtering_visualization(filtered_events, k_sigma)
        filtered_events_MeanStd_train.append((filtered_events, label))
        Err_list_MeanStd.append(ERR_MStd)
    for event, label in test_dataset_raw:
        MeanStdFilter = MaskMeanStandardDeviation(event, scale_factor)
        filtered_events, ERR_MStd = MeanStdFilter.Mean_std_thresholding(k_sigma)
        # MeanStdFilter.MeanStd_filtering_visualization(filtered_events, k_sigma)
        filtered_events_MeanStd_test.append((filtered_events, label))
        Err_list_MeanStd.append(ERR_MStd)

    end_time_MeanStd = time.time()
    time_MeanStd = end_time_MeanStd - start_time_MeanStd

    average_ERR_MeanStd = sum(Err_list_MeanStd) / len(Err_list_MeanStd)
    print(f"\nAverage Mean-StdDev Filtering Error (ERR) across all events: {average_ERR_MeanStd:.4f}")
    print(f"time Mean-StdDev Filtering = {time_MeanStd:.2f} seconds")
    
    save_filtered_dataset(
        filtered_events_MeanStd_train,
        save_dir=f"{filtering_root}/Threshold_{k_sigma}/train",
        prefix="train"
    )

    save_filtered_dataset(
        filtered_events_MeanStd_test,
        save_dir=f"{filtering_root}/Threshold_{k_sigma}/MeanStd/test",
        prefix="test"
    )

    return average_ERR_MeanStd, time_MeanStd

def GlobalSaliency_filtering_pipeline(train_dataset_raw, test_dataset_raw, use_percentile, parameters, filtering_root = "FilterTuning", scale_factor=3):
    
    """ parameters: 
    if use_percentile = True, parameters is the percentile value; 
    if use_percentile = False, parameters is the threshold value """

    filtered_events_GlobalSaliencyCrop_train = []
    filtered_events_GlobalSaliencyCrop_test = []
    Err_list_GlobalSaliencyCrop = []

    #TODO sistemare qua e dividere tra percentile e threshold

    start_time_GlobalSaliencyCrop = time.time()
    for event, label in train_dataset_raw:
        # Initialize and run Global Saliency Based Cropping
        GlobalSaliencyCropper = MaskGlobalSaliencyBasedCropping(event, scale_factor)
        filtered_events, OMS_norm, cropped_OMS_map, crop_box, ERR_global = GlobalSaliencyCropper.MaskGlobalSaliency_filtering(use_percentile, parameters, parameters)
        # GlobalSaliencyCropper.MaskGlobalSaliency_filtering_visualization(cropped_OMS_map, OMS_norm)
        filtered_events_GlobalSaliencyCrop_train.append((filtered_events, label))
        Err_list_GlobalSaliencyCrop.append(ERR_global)  
    for event, label in test_dataset_raw:
        # Initialize and run Global Saliency Based Cropping
        GlobalSaliencyCropper = MaskGlobalSaliencyBasedCropping(event, scale_factor)
        filtered_events, OMS_norm, cropped_OMS_map, crop_box, ERR_global = GlobalSaliencyCropper.MaskGlobalSaliency_filtering(use_percentile, parameters, parameters)
        # GlobalSaliencyCropper.MaskGlobalSaliency_filtering_visualization(cropped_OMS_map, OMS_norm)
        filtered_events_GlobalSaliencyCrop_test.append((filtered_events, label))
        Err_list_GlobalSaliencyCrop.append(ERR_global) 
    end_time_GlobalSaliencyCrop = time.time()
    time_GlobalSaliencyCrop = end_time_GlobalSaliencyCrop - start_time_GlobalSaliencyCrop

    average_ERR_GlobalSaliencyCrop = sum(Err_list_GlobalSaliencyCrop) / len(Err_list_GlobalSaliencyCrop)
    print(f"\nAverage Global Saliency Crop Filtering Error (ERR) across all events: {average_ERR_GlobalSaliencyCrop:.4f}")
    print(f"time Global Saliency Crop Filtering = {time_GlobalSaliencyCrop:.2f} seconds")
    
    if use_percentile:
        save_dir_train = f"{filtering_root}/Percentile_{parameters}/train"
        save_dir_test = f"{filtering_root}/Percentile_{parameters}/test"
    else:
        save_dir_train = f"{filtering_root}/Threshold_{parameters}/train"
        save_dir_test = f"{filtering_root}/Threshold_{parameters}/test"

    save_filtered_dataset(
        filtered_events_GlobalSaliencyCrop_train,
        save_dir=save_dir_train,
        prefix="train"
    )

    save_filtered_dataset(
        filtered_events_GlobalSaliencyCrop_test,
        save_dir=save_dir_test,
        prefix="test"
    )


def evaluate_parameters(kp, filtered_root = "FilterTuning"):

    train_dataset = FilteredNPZDataset(f"{filtered_root}/Threshold_{kp}/train")
    test_dataset  = FilteredNPZDataset(f"{filtered_root}/Threshold_{kp}/test")
    acc, train_time = train_model(train_dataset, test_dataset)

    return acc, train_time

def parameter_tuning_pipeline(parameters, train_dataset_raw, test_dataset_raw, tecnique, filtered_root = "FilterEvaluationTuning"):

    results = {}

    for i in range(len(parameters)):

        param = parameters[i]
        if tecnique == "Goal Oriented Thresholding":
            print(f"\nEvaluating Goal Oriented Thresholding with keep percentage: {param}%")
            average_ERR_GoalOriented, time_GoalOriented = GoalOriented_filtering_pipeline(train_dataset_raw, test_dataset_raw, param, filtered_root)
        elif tecnique == "Mean Standard Deviation":
            print(f"\nEvaluating Mean Standard Deviation Thresholding with k_sigma: {param}")
            average_ERR_GoalOriented, time_GoalOriented = MeanStd_filtering_pipeline(train_dataset_raw, test_dataset_raw, param, filtered_root)
        elif tecnique == "Global Saliency Crop use percentile":
            print(f"\nEvaluating Global Saliency Crop with percentile: {param}%")
            average_ERR_GoalOriented, time_GoalOriented = GlobalSaliency_filtering_pipeline(train_dataset_raw, test_dataset_raw, True, param, filtered_root)
        elif tecnique == "Global Saliency Crop not use percentile":
            print(f"\nEvaluating Global Saliency Crop with threshold: {param}")
            average_ERR_GoalOriented, time_GoalOriented = GlobalSaliency_filtering_pipeline(train_dataset_raw, test_dataset_raw, False, param, filtered_root)
        else :
            raise ValueError("Invalid technique specified.")
        acc, train_time = evaluate_parameters(param, filtered_root)
        results[i] = {
            "threshold": param,
            "ERR": average_ERR_GoalOriented,
            "time Goal Oriented Filtering": time_GoalOriented,
            "accuracy": acc,
            "time": train_time
        }
        print(f"[Threshold {param}%] Accuracy: {acc:.2f}% | Time: {train_time:.1f}s")

        write_parameter_tuning_results_to_file(tecnique, param, average_ERR_GoalOriented, time_GoalOriented, acc, train_time)

    return results

def plot_threshold_vs_accuracy(results, parameters):
    accuracies = [results[k]["accuracy"] for k in parameters]

    plt.figure(figsize=(7,5))
    plt.plot(parameters, accuracies, "o-", linewidth=2)
    plt.xlabel("OMS Keep Percentage (%)")
    plt.ylabel("Classification Accuracy (%)")
    plt.title("Goal-Oriented OMS Threshold Tuning")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":

    # ---- DVSGesture Dataset -----
    print("Loading downsampled DVSGesture dataset...")

    training_ROOT = "C:/Users/giuli/Desktop/Giulia/PER/Event-Based-Project/DVSGestureDownsampled/ibmGestureTrain"
    testing_ROOT = "C:/Users/giuli/Desktop/Giulia/PER/Event-Based-Project/DVSGestureDownsampled/ibmGestureTest"

    #training_ROOT = "C:/Users/giuli/Desktop/Giulia/PER/Event-Based-Project/Datasets/ibmGestureTrain"
    #testing_ROOT = "C:/Users/giuli/Desktop/Giulia/PER/Event-Based-Project/Datasets/ibmGestureTest"

    training_users = sorted(os.listdir(training_ROOT))
    test_users = sorted(os.listdir(testing_ROOT))

    train_dataset_raw = DVSGestureNPYDataset(training_ROOT, users=training_users)
    test_dataset_raw = DVSGestureNPYDataset(testing_ROOT, users=test_users)

    # Parameter tuning for Goal Oriented Thresholding
    parameters_GoalOrientedThresholding = [1, 2, 5, 10, 20, 30, 40]
    # parameters_GoalOrientedThresholding = [5] # Keep percentage values to evaluate (e.g., 1%, 5%, 10%, ..., 50%)
 
    result_GoalOrientedThresholding = parameter_tuning_pipeline(parameters_GoalOrientedThresholding, train_dataset_raw, test_dataset_raw, tecnique="Goal Oriented Thresholding", filtered_root="FilteredGoalOrientedTuning")
    

    # Parameter tuning for Mean Standard Deviation Thresholding
    parameters_MeanStd = [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]

    results_MeanStd = parameter_tuning_pipeline(parameters_MeanStd, train_dataset_raw, test_dataset_raw, tecnique="Mean Standard Deviation", filtered_root="FilteredMeanStdTuning")

    # Parameter tuning for Global Saliency-based Cropping
    parameters_percentile_GlobalSaliency = [85, 88, 90, 92, 95]
    parameters_thresholds_GlobalSaliency = [0.15, 0.20, 0.25]

    results_GlobalSaliency_percentile = parameter_tuning_pipeline(parameters_percentile_GlobalSaliency, train_dataset_raw, test_dataset_raw, tecnique="Global Saliency Crop use percentile", filtered_root="FilteredGlobalSaliencyPercentileTuning")
    
    results_GlobalSaliency_threshold = parameter_tuning_pipeline(parameters_thresholds_GlobalSaliency, train_dataset_raw, test_dataset_raw, tecnique="Global Saliency Crop not use percentile", filtered_root="FilteredGlobalSaliencyThresholdTuning")
    
    plot_threshold_vs_accuracy(result_GoalOrientedThresholding, parameters_GoalOrientedThresholding)
    plot_threshold_vs_accuracy(results_MeanStd, parameters_MeanStd)
    plot_threshold_vs_accuracy(results_GlobalSaliency_percentile, parameters_percentile_GlobalSaliency)
    plot_threshold_vs_accuracy(results_GlobalSaliency_threshold, parameters_thresholds_GlobalSaliency)







