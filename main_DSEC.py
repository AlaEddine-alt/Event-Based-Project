import numpy as np
import time
import os

from Filtering_techniques.OMSSaliencyMapFiltering import OMSFiltering
from Filtering_techniques.AttentionMapFiltering import AttentionFiltering
from Filtering_techniques.MaskAdaptiveElbow import AdaptiveElbowOMSFiltering
from Filtering_techniques.MaskGoalOriented import MaskGoalOrientedOMSFiltering
from Filtering_techniques.MaskMeanStandardDeviation import MaskMeanStandardDeviation
from Filtering_techniques.MaskGlobalSaliencyBasedCropping import MaskGlobalSaliencyBasedCropping
from Filtering_techniques.Denoise import Denoise
from Filtering_techniques.RandomCropFiltering import RandomCropFiltering
from functions.writeResultsFunctions import write_filtering_results_to_file


def load_dsec_events(file_path):
    """
    Load events from DSEC txt file.
    
    Args:
        file_path: Path to the DSEC.txt file
    
    Returns:
        event: Dictionary with keys 'x', 'y', 't', 'p'
        scale_factor: Scale factor for DSEC (0.75)
    """
    print(f"Loading DSEC events from {file_path}...")
    
    data = np.loadtxt(file_path)
    
    xs = data[:, 0].astype(int)
    ys = data[:, 1].astype(int)
    timestamps = data[:, 2]
    pols = data[:, 3].astype(int)
    
    event = {
        'x': xs,
        'y': ys,
        't': timestamps,
        'p': pols
    }
    
    scale_factor = 0.75  
    
    print(f"Loaded {len(xs)} events")
    print(f"Event shape - X range: [{xs.min()}, {xs.max()}], Y range: [{ys.min()}, {ys.max()}]")
    
    return event, scale_factor


if __name__ == "__main__":

    # ---- DSEC Dataset -----
    print("DSEC Filtering Analysis")
    
    # Load the DSEC event data
    dsec_file_path = "Datasets/Dsec.txt"
    event, scale_factor = load_dsec_events(dsec_file_path)
    
    threshold_OMS = 0.2 
    
    # --- OMS Filtering ---
    print("OMS Filtering")
    
    start_time_OMS = time.time()
    OMSfilter = OMSFiltering(event, scale_factor, threshold_OMS)
    OMSMap, filtered_event_OMS, I_filtered, Err_OMS = OMSfilter.OMS_filtering()
    end_time_OMS = time.time()
    time_OMS = end_time_OMS - start_time_OMS
    
    print(f"OMS Event Reduction Ratio (ERR): {Err_OMS:.4f}")
    print(f"Time OMS filtering: {time_OMS:.2f} seconds")
    
    
    # --- Attention Filtering ---
    print("Attention Filtering")
    
    start_time_Attention = time.time()
    Attentionfilter = AttentionFiltering(event, scale_factor)
    filtered_event_Attention, saliency_map, Err_Attention = Attentionfilter.Attention_filtering()
    end_time_Attention = time.time()
    time_Attention = end_time_Attention - start_time_Attention
    
    print(f"Attention Filtering Event Reduction Ratio (ERR): {Err_Attention:.4f}")
    print(f"Time Attention filtering: {time_Attention:.2f} seconds")
    
    
    # --- Adaptive Elbow Thresholding ---
    print("Adaptive Elbow Thresholding Filtering")
    
    start_time_adaptiveElbow = time.time()
    AdaptiveElbowFilter = AdaptiveElbowOMSFiltering(event, scale_factor, threshold_OMS)
    filtered_events, masked_OMS, OMSMap, Err_adElow = AdaptiveElbowFilter.Albowdaptive_thresholding()
    end_time_adaptiveElbow = time.time()
    time_adaptiveElbow = end_time_adaptiveElbow - start_time_adaptiveElbow
    
    print(f"Adaptive Elbow Event Reduction Ratio (ERR): {Err_adElow:.4f}")
    print(f"Time Adaptive Elbow filtering: {time_adaptiveElbow:.2f} seconds")
    
    
    # --- Goal Oriented Thresholding ---
    print("Goal Oriented Thresholding Filtering")
    
    keep_percent = 30  
    
    start_time_GoalOriented = time.time()
    GoalOrientedFilter = MaskGoalOrientedOMSFiltering(event, scale_factor, threshold_OMS)
    filtered_events, masked_OMS, OMSMap, ERR_goal = GoalOrientedFilter.Goadaptive_thresholding(keep_percent)
    end_time_GoalOriented = time.time()
    time_GoalOriented = end_time_GoalOriented - start_time_GoalOriented
    
    print(f"Goal-Oriented Event Reduction Ratio (ERR): {ERR_goal:.4f}")
    print(f"Time Goal Oriented Filtering: {time_GoalOriented:.2f} seconds")
    
    
    # --- Mean and Standard Deviation Thresholding ---
    print("Mean and Standard Deviation Thresholding Filtering")
    
    k_sigma = 0.75 
    
    start_time_MeanStd = time.time()
    MeanStdFilter = MaskMeanStandardDeviation(event, scale_factor, threshold_OMS)
    filtered_events, ERR_MStd = MeanStdFilter.Mean_std_thresholding(k_sigma)
    end_time_MeanStd = time.time()
    time_MeanStd = end_time_MeanStd - start_time_MeanStd
    
    print(f"Mean-StdDev Event Reduction Ratio (ERR): {ERR_MStd:.4f}")
    print(f"Time Mean-StdDev Filtering: {time_MeanStd:.2f} seconds")
    
    
    # --- Global Saliency Based Cropping ---
    print("Global Saliency Based Cropping Filtering")
    
    Use_percentile = True
    percentile = 85 
    threshold = 0.15
    
    start_time_GlobalSaliencyCrop = time.time()
    GlobalSaliencyCropper = MaskGlobalSaliencyBasedCropping(event, scale_factor, threshold_OMS)
    filtered_events, OMS_norm, cropped_OMS_map, crop_box, ERR_global = GlobalSaliencyCropper.MaskGlobalSaliency_filtering(Use_percentile, percentile, threshold)
    end_time_GlobalSaliencyCrop = time.time()
    time_GlobalSaliencyCrop = end_time_GlobalSaliencyCrop - start_time_GlobalSaliencyCrop
    
    print(f"Global Saliency Crop Event Reduction Ratio (ERR): {ERR_global:.4f}")
    print(f"Time Global Saliency Crop Filtering: {time_GlobalSaliencyCrop:.2f} seconds")
    
    
    # --- Denoising Filtering ---
    print("Denoising Filtering")
    
    start_time_Denoised = time.time()
    DenoiseFiltering = Denoise(event, scale_factor)
    events_denoised, ERR = DenoiseFiltering.Denoise_filtering()
    end_time_Denoised = time.time()
    time_Denoised = end_time_Denoised - start_time_Denoised
    
    print(f"Denoising Event Reduction Ratio (ERR): {ERR:.4f}")
    print(f"Time Denoising Filtering: {time_Denoised:.2f} seconds")
    
    
    # --- Random Cropping Filtering ---
    print("Random Cropping Filtering")
    
    sensor_size = (None, None)  # Will be determined from data dimensions
    crop_size = (64, 64)
    
    # Auto-detect sensor size from event coordinates
    max_x = int(np.max(event['x'])) + 1
    max_y = int(np.max(event['y'])) + 1
    sensor_size = (max_y, max_x)
    
    start_time_RandomCrop = time.time()
    RandomCrop_filtering = RandomCropFiltering(event, scale_factor, sensor_size, crop_size)
    events_cropped, ERR_crop = RandomCrop_filtering.RandomCrop_filtering(sensor_size, crop_size)
    end_time_RandomCrop = time.time()
    time_RandomCrop = end_time_RandomCrop - start_time_RandomCrop
    
    print(f"Random Crop Event Reduction Ratio (ERR): {ERR_crop:.4f}")
    print(f"Time Random Crop Filtering: {time_RandomCrop:.2f} seconds")
    
    
    # --- Summary ---
    print("SUMMARY - DSEC Filtering Results")
    
    results = {
        "OMS Filtering": (Err_OMS, time_OMS),
        "Attention Filtering": (Err_Attention, time_Attention),
        "Adaptive Elbow Thresholding": (Err_adElow, time_adaptiveElbow),
        "Goal Oriented Thresholding": (ERR_goal, time_GoalOriented),
        "Mean-StdDev Thresholding": (ERR_MStd, time_MeanStd),
        "Global Saliency Based Cropping": (ERR_global, time_GlobalSaliencyCrop),
        "Denoising Filtering": (ERR, time_Denoised),
        "Random Crop Filtering": (ERR_crop, time_RandomCrop)
    }
    
    print(f"\n{'Filtering Technique':<40} {'ERR':<10} {'Time (s)':<10}")
    print("-" * 60)
    for technique, (err, elapsed_time) in results.items():
        print(f"{technique:<40} {err:<10.4f} {elapsed_time:<10.2f}")
