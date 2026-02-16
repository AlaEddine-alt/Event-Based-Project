import tonic
import time
import os

from functions.loadDatasetFunctions import load_events, DVSGestureNPYDataset
from Filtering_techniques.OMSSaliencyMapFiltering import OMSFiltering
from Filtering_techniques.AttentionMapFiltering import AttentionFiltering
from Filtering_techniques.MaskAdaptiveElbow  import AdaptiveElbowOMSFiltering
from Filtering_techniques.MaskGoalOriented import MaskGoalOrientedOMSFiltering
from Filtering_techniques.MaskMeanStandardDeviation import MaskMeanStandardDeviation
from Filtering_techniques.MaskGlobalSaliencyBasedCropping import MaskGlobalSaliencyBasedCropping
from Filtering_techniques.TecniquesNotUsed.Random_filtering import RandomEventFiltering
from Filtering_techniques.Denoise import Denoise
from Filtering_techniques.RandomCropFiltering import RandomCropFiltering
from Classification.ComplexCNN import train_model
from functions.saveAndLoadFilteredData import save_filtered_dataset
from functions.writeResultsFunctions import write_filtering_results_to_file


if __name__ == "__main__":

    # ---- DVSGesture Dataset -----
    print("Loading DVSGesture dataset...")

    # training_ROOT = "C:/Users/giuli/Desktop/Giulia/PER/Event-Based-Project/DVSGestureDownsampled/ibmGestureTrain"
    # testing_ROOT = "C:/Users/giuli/Desktop/Giulia/PER/Event-Based-Project/DVSGestureDownsampled/ibmGestureTest"

    training_ROOT = "C:/Users/giuli/Desktop/Giulia/PER/Event-Based-Project/Datasets/ibmGestureTrain"
    testing_ROOT = "C:/Users/giuli/Desktop/Giulia/PER/Event-Based-Project/Datasets/ibmGestureTest"

    training_users = sorted(os.listdir(training_ROOT))
    test_users = sorted(os.listdir(testing_ROOT))

    train_dataset_raw = DVSGestureNPYDataset(training_ROOT, users=training_users)
    test_dataset_raw = DVSGestureNPYDataset(testing_ROOT, users=test_users)

    scale_factor = 3
    
    # --- OMS Filtering ---
    
    filtered_events_OMS_train = []
    filtered_events_OMS_test = []
    Err_list_OMS = []
    threshold_OMS = 0.3 
    
    start_time_OMS = time.time()
    for event, label in train_dataset_raw:
        print("processing event")
        # Initialize and run OMS Filtering
        OMSfilter = OMSFiltering(event, scale_factor, threshold_OMS)
        OMSMap, filtered_event_OMS, I_filtered, Err_OMS = OMSfilter.OMS_filtering()
        # OMSfilter.OMS_filtering_visualization(OMSMap, I_filtered)
        filtered_events_OMS_train.append((filtered_event_OMS, label))
        Err_list_OMS.append(Err_OMS)
    for event, label in test_dataset_raw:
        # Initialize and run OMS Filtering
        OMSfilter = OMSFiltering(event, scale_factor, threshold_OMS)
        OMSMap, filtered_event_OMS, I_filtered, Err_OMS = OMSfilter.OMS_filtering()
        # OMSfilter.OMS_filtering_visualization(OMSMap, I_filtered)
        filtered_events_OMS_test.append((filtered_event_OMS, label))
        Err_list_OMS.append(Err_OMS)
    end_time_OMS = time.time()
    time_OMS = end_time_OMS - start_time_OMS

    average_ERR_OMS = sum(Err_list_OMS) / len(Err_list_OMS)
    print(f"\nAverage OMS Event Reduction Ratio (ERR) across all events: {average_ERR_OMS:.4f}")
    print(f"time OMS filtering = {time_OMS:.2f} seconds")
    
    save_filtered_dataset(
        filtered_events_OMS_train,
        save_dir="Datasets/FilteredDatasets/OMS/train",
        prefix="train"
    )

    save_filtered_dataset(
        filtered_events_OMS_test,
        save_dir="Datasets/FilteredDatasets/OMS/test",
        prefix="test"
    )

    write_filtering_results_to_file("OMS Filtering", average_ERR_OMS, time_OMS)
    
    
    # --- Attention Filtering ---
    
    filtered_events_Attention_train = []
    filtered_events_Attention_test = []
    Err_list_Attention = []
    
    start_time_Attention = time.time()
    for event, label in train_dataset_raw:
        # Initialize and run Attention Filtering
        Attentionfilter = AttentionFiltering(event, scale_factor)
        filtered_event_Attention, saliency_map, Err_Attention = Attentionfilter.Attention_filtering()
        # Attentionfilter.Attention_visualization(saliency_map)
        filtered_events_Attention_train.append((filtered_event_Attention, label))
        Err_list_Attention.append(Err_Attention)
    for event, label in test_dataset_raw:
        # Initialize and run Attention Filtering
        Attentionfilter = AttentionFiltering(event, scale_factor)
        filtered_event_Attention, saliency_map, Err_Attention = Attentionfilter.Attention_filtering()
        # Attentionfilter.Attention_visualization(saliency_map)
        filtered_events_Attention_test.append((filtered_event_Attention, label))
        Err_list_Attention.append(Err_Attention)
    end_time_Attention = time.time()
    time_Attention = end_time_Attention - start_time_Attention

    average_ERR_Attention = sum(Err_list_Attention) / len(Err_list_Attention)
    print(f"\nAverage Attention filtering Event Reduction Ratio (ERR) across all events: {average_ERR_Attention:.4f}")
    print(f"time Attention filtering = {time_Attention:.2f} seconds")
    
    save_filtered_dataset(
        filtered_events_Attention_train,
        save_dir="Datasets/FilteredDatasets/Attention/train",
        prefix="train"
    )

    save_filtered_dataset(
        filtered_events_Attention_test,
        save_dir="Datasets/FilteredDatasets/Attention/test",
        prefix="test"
    )

    write_filtering_results_to_file("Attention Filtering", average_ERR_Attention, time_Attention)
    
    
    # --- Adaptive Elbow Thresholding ---
    
    filtered_events_adaptiveElbow_train = []
    filtered_events_adaptiveElbow_test = []
    Err_list_adaptiveElbow = []

    start_time_adaptiveElbow = time.time()
    for event, label in train_dataset_raw:
        # Initialize and run Adaptive Elbow Thresholding
        AdaptiveElbowFilter = AdaptiveElbowOMSFiltering(event, scale_factor, threshold_OMS)
        filtered_events, masked_OMS, OMSMap, Err_adElow = AdaptiveElbowFilter.Albowdaptive_thresholding()
        # AdaptiveElbowFilter.AdaptiveElbow_filtering_visualization(OMSMap, masked_OMS)
        filtered_events_adaptiveElbow_train.append((filtered_events, label))
        Err_list_adaptiveElbow.append(Err_adElow)
    
    for event, label in test_dataset_raw:
        # Initialize and run Adaptive Elbow Thresholding
        AdaptiveElbowFilter = AdaptiveElbowOMSFiltering(event, scale_factor, threshold_OMS)
        filtered_events, masked_OMS, OMSMap, Err_adElow = AdaptiveElbowFilter.Albowdaptive_thresholding()
        # AdaptiveElbowFilter.AdaptiveElbow_filtering_visualization(OMSMap, masked_OMS)
        filtered_events_adaptiveElbow_test.append((filtered_events, label))
        Err_list_adaptiveElbow.append(Err_adElow)

    end_time_adaptiveElbow = time.time()
    time_adaptiveElbow = end_time_adaptiveElbow - start_time_adaptiveElbow

    average_ERR_adaptiveElbow = sum(Err_list_adaptiveElbow) / len(Err_list_adaptiveElbow)
    print(f"\nAverage Event Reduction Ratio (ERR) across all events: {average_ERR_adaptiveElbow:.4f}")
    print(f"time Adaptive Elbow filtering = {time_adaptiveElbow:.2f} seconds")

    save_filtered_dataset(
        filtered_events_adaptiveElbow_train,
        save_dir="Datasets/FilteredDatasets/AdaptiveElbow/train",
        prefix="train"
    )

    save_filtered_dataset(
        filtered_events_adaptiveElbow_test,
        save_dir="Datasets/FilteredDatasets/AdaptiveElbow/test",
        prefix="test"
    )

    write_filtering_results_to_file("Adaptive Elbow Thresholding", average_ERR_adaptiveElbow, time_adaptiveElbow)
    
    
    # --- Goal Oriented Thresholding ---
    
    filtered_events_GoalOriented_train = []
    filtered_events_GoalOriented_test = []
    Err_list_GoalOriented = []
    keep_percent = 5  # Keep top 5%
    
    start_time_GoalOriented = time.time()
    for event, label in train_dataset_raw:
        print("processing")
        # Initialize and run Goal Oriented Thresholding
        GoalOrientedFilter = MaskGoalOrientedOMSFiltering(event, scale_factor, threshold_OMS)
        filtered_events, masked_OMS, OMSMap, ERR_goal = GoalOrientedFilter.Goadaptive_thresholding(keep_percent) 
        # GoalOrientedFilter.GoalOriented_filtering_visualization(OMSMap, masked_OMS)
        filtered_events_GoalOriented_train.append((filtered_events, label))
        Err_list_GoalOriented.append(ERR_goal)
    for event, label in test_dataset_raw:
        # Initialize and run Goal Oriented Thresholding
        GoalOrientedFilter = MaskGoalOrientedOMSFiltering(event, scale_factor, threshold_OMS)
        filtered_events, masked_OMS, OMSMap, ERR_goal = GoalOrientedFilter.Goadaptive_thresholding(keep_percent) 
        # GoalOrientedFilter.GoalOriented_filtering_visualization(OMSMap, masked_OMS)
        filtered_events_GoalOriented_test.append((filtered_events, label))
        Err_list_GoalOriented.append(ERR_goal)
    end_time_GoalOriented = time.time()
    time_GoalOriented = end_time_GoalOriented - start_time_GoalOriented

    average_ERR_GoalOriented = sum(Err_list_GoalOriented) / len(Err_list_GoalOriented)
    print(f"\nAverage Goal-Oriented Event Reduction Ratio (ERR) across all events: {average_ERR_GoalOriented:.4f}")
    print(f"time Goal Oriented Filtering = {time_GoalOriented:.2f} seconds")

    save_filtered_dataset(
        filtered_events_GoalOriented_train,
        save_dir="Datasets/FilteredDatasets/GoalOrientedThresholding/train",
        prefix="train"
    )

    save_filtered_dataset(
        filtered_events_GoalOriented_test,
        save_dir="Datasets/FilteredDatasets/GoalOrientedThresholding/test",
        prefix="test"
    )

    write_filtering_results_to_file("Goal Oriented Thresholding", average_ERR_GoalOriented, time_GoalOriented)
    

    # --- Mean and Standard Deviation Thresholding ---
    
    filtered_events_MeanStd_train = []
    filtered_events_MeanStd_test = []
    Err_list_MeanStd = []
    k_sigma = 2.0  # Parameter for thresholding
    
    start_time_MeanStd = time.time()
    # Initialize and run Mean and Standard Deviation Thresholding
    for event, label in train_dataset_raw:
        print(f"Processing event ")
        MeanStdFilter = MaskMeanStandardDeviation(event, scale_factor, threshold_OMS)
        filtered_events, ERR_MStd = MeanStdFilter.Mean_std_thresholding(k_sigma)
        # MeanStdFilter.MeanStd_filtering_visualization(filtered_events, k_sigma)
        filtered_events_MeanStd_train.append((filtered_events, label))
        Err_list_MeanStd.append(ERR_MStd)
    for event, label in test_dataset_raw:
        MeanStdFilter = MaskMeanStandardDeviation(event, scale_factor, threshold_OMS)
        filtered_events, ERR_MStd = MeanStdFilter.Mean_std_thresholding(k_sigma)
        # MeanStdFilter.MeanStd_filtering_visualization(filtered_events, k_sigma)
        filtered_events_MeanStd_test.append((filtered_events, label))
        Err_list_MeanStd.append(ERR_MStd)

    end_time_MeanStd = time.time()
    time_MeanStd = end_time_MeanStd - start_time_MeanStd

    average_ERR_MeanStd = sum(Err_list_MeanStd) / len(Err_list_MeanStd)
    print(f"\nAverage Mean-StdDev Event Reduction Ratio (ERR) across all events: {average_ERR_MeanStd:.4f}")
    print(f"time Mean-StdDev Filtering = {time_MeanStd:.2f} seconds")
    
    save_filtered_dataset(
        filtered_events_MeanStd_train,
        save_dir="Datasets/FilteredDatasets/MeanStd/train",
        prefix="train"
    )

    save_filtered_dataset(
        filtered_events_MeanStd_test,
        save_dir="Datasets/FilteredDatasets/MeanStd/test",
        prefix="test"
    )

    write_filtering_results_to_file("Mean-StdDev Thresholding", average_ERR_MeanStd, time_MeanStd)
    
    
    # --- Global Saliency Based Cropping ---
    
    filtered_events_GlobalSaliencyCrop_train = []
    filtered_events_GlobalSaliencyCrop_test = []
    Err_list_GlobalSaliencyCrop = []
    Use_percentile = True
    percentile = 90
    threshold = 0.4

    start_time_GlobalSaliencyCrop = time.time()
    for event, label in train_dataset_raw:
        # Initialize and run Global Saliency Based Cropping
        GlobalSaliencyCropper = MaskGlobalSaliencyBasedCropping(event, scale_factor, threshold_OMS)
        filtered_events, OMS_norm, cropped_OMS_map, crop_box, ERR_global = GlobalSaliencyCropper.MaskGlobalSaliency_filtering(Use_percentile, percentile, threshold)
        # GlobalSaliencyCropper.MaskGlobalSaliency_filtering_visualization(cropped_OMS_map, OMS_norm)
        filtered_events_GlobalSaliencyCrop_train.append((filtered_events, label))
        Err_list_GlobalSaliencyCrop.append(ERR_global)  
    for event, label in test_dataset_raw:
        # Initialize and run Global Saliency Based Cropping
        GlobalSaliencyCropper = MaskGlobalSaliencyBasedCropping(event, scale_factor, threshold_OMS)
        filtered_events, OMS_norm, cropped_OMS_map, crop_box, ERR_global = GlobalSaliencyCropper.MaskGlobalSaliency_filtering(Use_percentile, percentile, threshold)
        # GlobalSaliencyCropper.MaskGlobalSaliency_filtering_visualization(cropped_OMS_map, OMS_norm)
        filtered_events_GlobalSaliencyCrop_test.append((filtered_events, label))
        Err_list_GlobalSaliencyCrop.append(ERR_global) 
    end_time_GlobalSaliencyCrop = time.time()
    time_GlobalSaliencyCrop = end_time_GlobalSaliencyCrop - start_time_GlobalSaliencyCrop

    average_ERR_GlobalSaliencyCrop = sum(Err_list_GlobalSaliencyCrop) / len(Err_list_GlobalSaliencyCrop)
    print(f"\nAverage Global Saliency Crop Event Reduction Ratio (ERR) across all events: {average_ERR_GlobalSaliencyCrop:.4f}")
    print(f"time Global Saliency Crop Filtering = {time_GlobalSaliencyCrop:.2f} seconds")
    
    save_filtered_dataset(
        filtered_events_GlobalSaliencyCrop_train,
        save_dir="Datasets/FilteredDatasets/GlobalSaliencyBasedCropping/train",
        prefix="train"
    )

    save_filtered_dataset(
        filtered_events_GlobalSaliencyCrop_test,
        save_dir="Datasets/FilteredDatasets/GlobalSaliencyBasedCropping/test",
        prefix="test"
    )

    write_filtering_results_to_file("Global Saliency Based Cropping", average_ERR_GlobalSaliencyCrop, time_GlobalSaliencyCrop)

    """
    # --- Denoising Filtering ---
    
    filtered_events_Denoised_train = []
    filtered_events_Denoised_test = []
    Err_list_Denoised = []

    start_time_Denoised = time.time()
    for event, label in train_dataset_raw:
        # Initialize and run Denoising Filtering
        DenoiseFiltering = Denoise(event, scale_factor)
        events_denoised, ERR = DenoiseFiltering.Denoise_filtering()
        # DenoiseFiltering.Denoise_filtering_visualization()
        filtered_events_Denoised_train.append((events_denoised, label))
        Err_list_Denoised.append(ERR)

    for event, label in test_dataset_raw:
        # Initialize and run Denoising Filtering
        DenoiseFiltering = Denoise(event, scale_factor)
        events_denoised, ERR = DenoiseFiltering.Denoise_filtering()
        # DenoiseFiltering.Denoise_filtering_visualization()
        filtered_events_Denoised_test.append((events_denoised, label))
        Err_list_Denoised.append(ERR)

    end_time_Denoised = time.time()
    time_Denoised = end_time_Denoised - start_time_Denoised

    average_ERR_Denoised = sum(Err_list_Denoised) / len(Err_list_Denoised)
    print(f"\nAverage Denoising Event Reduction Ratio (ERR) across all events: {average_ERR_Denoised:.4f}")
    print(f"time Denoising Filtering = {time_Denoised:.2f} seconds")
    
    save_filtered_dataset(
        filtered_events_Denoised_train,
        save_dir="Datasets/FilteredDatasets/Denoise/train",
        prefix="train"
    )

    save_filtered_dataset(
        filtered_events_Denoised_test,
        save_dir="Datasets/FilteredDatasets/Denoise/test",
        prefix="test"
    )
    
    write_filtering_results_to_file("Denoising Filtering", average_ERR_Denoised, time_Denoised) 
    """

    # --- Random Cropping Filtering ---

    filtered_events_RandomCrop_train = []
    filtered_events_RandomCrop_test = []
    Err_list_RandomCrop = []
    sensor_size = (128, 128)      # original DVS resolution
    crop_size = (64, 64)         # desired crop size

    start_time_RandomCrop = time.time()

    for event, label in train_dataset_raw:
        # Initialize and run Random Cropping Filtering
        RandomCrop_filtering = RandomCropFiltering(event, scale_factor, sensor_size, crop_size)
        events_cropped, ERR_crop = RandomCrop_filtering.RandomCrop_filtering(sensor_size, crop_size)
        # RandomCrop_filtering.RandomCrop_filtering_visualization(events_cropped)
        filtered_events_RandomCrop_train.append((events_cropped, label))
        Err_list_RandomCrop.append(ERR_crop)

    for event, label in test_dataset_raw:
        # Initialize and run Random Cropping Filtering
        RandomCrop_filtering = RandomCropFiltering(event, scale_factor, sensor_size, crop_size)
        events_cropped, ERR_crop = RandomCrop_filtering.RandomCrop_filtering(sensor_size, crop_size)
        # RandomCrop_filtering.RandomCrop_filtering_visualization(events_cropped)
        filtered_events_RandomCrop_test.append((events_cropped, label))
        Err_list_RandomCrop.append(ERR_crop)

    end_time_RandomCrop = time.time()
    time_RandomCrop = end_time_RandomCrop - start_time_RandomCrop

    average_ERR_RandomCrop = sum(Err_list_RandomCrop) / len(Err_list_RandomCrop)    
    print(f"\nAverage Random Crop Event Reduction Ratio (ERR) across all events: {average_ERR_RandomCrop:.4f}")   
    print(f"time Random Crop Filtering = {time_RandomCrop:.2f} seconds")

    save_filtered_dataset(
        filtered_events_RandomCrop_train,
        save_dir="Datasets/FilteredDatasets/RandomCrop/train",
        prefix="train"
    )

    save_filtered_dataset(
        filtered_events_RandomCrop_test,
        save_dir="Datasets/FilteredDatasets/RandomCrop/test",
        prefix="test"
    )
    
    write_filtering_results_to_file("Random Crop Filtering", average_ERR_RandomCrop, time_RandomCrop)






