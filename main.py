import tonic
import time

from functions.loadDatasetFunctions import load_events
from Filtering_techniques.OMSSaliencyMapFiltering import OMSFiltering
from Filtering_techniques.MaskAdaptiveElbow  import AdaptiveElbowOMSFiltering
from Filtering_techniques.MaskGoalOriented import MaskGoalOrientedOMSFiltering
from Filtering_techniques.MaskMeanStandardDeviation import MaskMeanStandardDeviation
from Filtering_techniques.MaskGlobalSaliencyBasedCropping import MaskGlobalSaliencyBasedCropping
from Filtering_techniques.TecniquesNotUsed.Random_filtering import RandomEventFiltering
from Filtering_techniques.Denoise import Denoise
from Filtering_techniques.RandomCropFiltering import RandomCropFiltering
from Classification.ComplexCNN import train_model


if __name__ == "__main__":

    # ---- DVSGesture Dataset -----
    event_list, target_list, scale_factor = load_events("DVSGesture")
    
    # --- OMS Filtering ---
    """
    filtered_events_OMS = []
    Err_list_OMS = []
    
    start_time_OMS = time.time()
    for event in event_list:
        # Initialize and run OMS Filtering
        OMSfilter = OMSFiltering(event, scale_factor)
        OMSMap, I_filtered, Err_OMS = OMSfilter.OMS_filtering()
        # OMSfilter.OMS_filtering_visualization(OMSMap, I_filtered)
        filtered_events_OMS.append(OMSMap)
        Err_list_OMS.append(Err_OMS)
    end_time_OMS = time.time()
    time_OMS = end_time_OMS - start_time_OMS

    average_ERR_OMS = sum(Err_list_OMS) / len(Err_list_OMS)
    print(f"\nAverage OMS Filtering Error (ERR) across all events: {average_ERR_OMS:.4f}")
    print(f"time OMS filtering = {time_OMS:.2f} seconds")
    """
    # --- Adaptive Elbow Thresholding ---
    
    filtered_events_adaptiveElbow = []
    Err_list_adaptiveElbow = []

    start_time_adaptiveElbow = time.time()
    for event in event_list:
        # Initialize and run Adaptive Elbow Thresholding
        AdaptiveElbowFilter = AdaptiveElbowOMSFiltering(event, scale_factor)
        filtered_events, masked_OMS, OMSMap, Err_adElow = AdaptiveElbowFilter.Albowdaptive_thresholding()
        # AdaptiveElbowFilter.AdaptiveElbow_filtering_visualization(OMSMap, masked_OMS)
        filtered_events_adaptiveElbow.append(filtered_events)
        Err_list_adaptiveElbow.append(Err_adElow)
    end_time_adaptiveElbow = time.time()
    time_adaptiveElbow = end_time_adaptiveElbow - start_time_adaptiveElbow

    average_ERR_adaptiveElbow = sum(Err_list_adaptiveElbow) / len(Err_list_adaptiveElbow)
    print(f"\nAverage Filtering Error (ERR) across all events: {average_ERR_adaptiveElbow:.4f}")
    print(f"time Adaptive Elbow filtering = {time_adaptiveElbow:.2f} seconds")
    

    # --- Goal Oriented Thresholding ---
    """
    filtered_events_GoalOriented = []
    Err_list_GoalOriented = []
    keep_percent = 5  # Keep top 5%
    
    start_time_GoalOriented = time.time()
    for event in event_list:
        # Initialize and run Goal Oriented Thresholding
        GoalOrientedFilter = MaskGoalOrientedOMSFiltering(event, scale_factor)
        filtered_events, masked_OMS, OMSMap, ERR_goal = GoalOrientedFilter.Goadaptive_thresholding(keep_percent) 
        # GoalOrientedFilter.GoalOriented_filtering_visualization(OMSMap, masked_OMS)
        filtered_events_GoalOriented.append(filtered_events)
        Err_list_GoalOriented.append(ERR_goal)
    end_time_GoalOriented = time.time()
    time_GoalOriented = end_time_GoalOriented - start_time_GoalOriented

    average_ERR_GoalOriented = sum(Err_list_GoalOriented) / len(Err_list_GoalOriented)
    print(f"\nAverage Goal-Oriented Filtering Error (ERR) across all events: {average_ERR_GoalOriented:.4f}")
    print(f"time Goal Oriented Filtering = {time_GoalOriented:.2f} seconds")
    """

    # --- Mean and Standard Deviation Thresholding ---
    """
    filtered_events_MeanStd = []
    Err_list_MeanStd = []
    k_sigma = 2.0  # Parameter for thresholding
    
    start_time_MeanStd = time.time()
    # Initialize and run Mean and Standard Deviation Thresholding
    for event in event_list:
        MeanStdFilter = MaskMeanStandardDeviation(event, scale_factor)
        filtered_events, ERR_MStd = MeanStdFilter.Mean_std_thresholding(k_sigma)
        # MeanStdFilter.MeanStd_filtering_visualization(filtered_events, k_sigma)
        filtered_events_MeanStd.append(filtered_events)
        Err_list_MeanStd.append(ERR_MStd)

    end_time_MeanStd = time.time()
    time_MeanStd = end_time_MeanStd - start_time_MeanStd

    average_ERR_MeanStd = sum(Err_list_MeanStd) / len(Err_list_MeanStd)
    print(f"\nAverage Mean-StdDev Filtering Error (ERR) across all events: {average_ERR_MeanStd:.4f}")
    print(f"time Mean-StdDev Filtering = {time_MeanStd:.2f} seconds")
    """

    # --- Global Saliency Based Cropping ---
    """
    filtered_events_GlobalSaliencyCrop = []
    Err_list_GlobalSaliencyCrop = []
    Use_percentile = True
    percentile = 90
    threshold = 0.4

    start_time_GlobalSaliencyCrop = time.time()
    for event in event_list:
        # Initialize and run Global Saliency Based Cropping
        GlobalSaliencyCropper = MaskGlobalSaliencyBasedCropping(event, scale_factor)
        filtered_events, OMS_norm, cropped_OMS_map, crop_box, ERR_global = GlobalSaliencyCropper.MaskGlobalSaliency_filtering(Use_percentile, percentile, threshold)
        # GlobalSaliencyCropper.MaskGlobalSaliency_filtering_visualization(cropped_OMS_map, OMS_norm)
        filtered_events_GlobalSaliencyCrop.append(filtered_events)
        Err_list_GlobalSaliencyCrop.append(ERR_global)   
    end_time_GlobalSaliencyCrop = time.time()
    time_GlobalSaliencyCrop = end_time_GlobalSaliencyCrop - start_time_GlobalSaliencyCrop

    average_ERR_GlobalSaliencyCrop = sum(Err_list_GlobalSaliencyCrop) / len(Err_list_GlobalSaliencyCrop)
    print(f"\nAverage Global Saliency Crop Filtering Error (ERR) across all events: {average_ERR_GlobalSaliencyCrop:.4f}")
    print(f"time Global Saliency Crop Filtering = {time_GlobalSaliencyCrop:.2f} seconds")
    """

    # --- Denoising Filtering ---
    """
    filtered_events_Denoised = []
    Err_list_Denoised = []

    start_time_Denoised = time.time()
    for event in event_list:
        # Initialize and run Denoising Filtering
        DenoiseFiltering = Denoise(event, scale_factor)
        events_denoised, ERR = DenoiseFiltering.Denoise_filtering()
        # DenoiseFiltering.Denoise_filtering_visualization()
        filtered_events_Denoised.append(events_denoised)
        Err_list_Denoised.append(ERR)
    end_time_Denoised = time.time()
    time_Denoised = end_time_Denoised - start_time_Denoised

    average_ERR_Denoised = sum(Err_list_Denoised) / len(Err_list_Denoised)
    print(f"\nAverage Denoising Filtering Error (ERR) across all events: {average_ERR_Denoised:.4f}")
    print(f"time Denoising Filtering = {time_Denoised:.2f} seconds")
    """

    # --- Random Cropping Filtering ---

    filtered_events_RandomCrop = []
    Err_list_RandomCrop = []
    sensor_size = (128, 128)      # original DVS resolution
    crop_size = (64, 64)         # desired crop size

    start_time_RandomCrop = time.time()
    for event in event_list:
        # Initialize and run Random Cropping Filtering
        RandomCrop_filtering = RandomCropFiltering(event, scale_factor, sensor_size, crop_size)
        events_cropped, ERR_crop = RandomCrop_filtering.RandomCrop_filtering(sensor_size, crop_size)
        # RandomCrop_filtering.RandomCrop_filtering_visualization(events_cropped)
        filtered_events_RandomCrop.append(events_cropped)
        Err_list_RandomCrop.append(ERR_crop)
    end_time_RandomCrop = time.time()
    time_RandomCrop = end_time_RandomCrop - start_time_RandomCrop

    average_ERR_RandomCrop = sum(Err_list_RandomCrop) / len(Err_list_RandomCrop)    
    print(f"\nAverage Random Crop Filtering Error (ERR) across all events: {average_ERR_RandomCrop:.4f}")   
    print(f"time Random Crop Filtering = {time_RandomCrop:.2f} seconds")

    """
    # Training the ComplexCNN model
    print("Loading DVSGesture dataset...")
    dataset_training = tonic.datasets.DVSGesture(save_to="../Datasets", train=True)
    dataset_testing = tonic.datasets.DVSGesture(save_to="../Datasets", train=False)

    train_model(dataset_training, dataset_testing)
    """



