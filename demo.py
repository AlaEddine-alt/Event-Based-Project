import tonic
import time
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import cv2

from Filtering_techniques.OMSSaliencyMapFiltering import OMSFiltering
from Filtering_techniques.AttentionMapFiltering import AttentionFiltering
from Filtering_techniques.MaskAdaptiveElbow import AdaptiveElbowOMSFiltering
from Filtering_techniques.MaskGoalOriented import MaskGoalOrientedOMSFiltering
from Filtering_techniques.MaskMeanStandardDeviation import MaskMeanStandardDeviation
from Filtering_techniques.MaskGlobalSaliencyBasedCropping import MaskGlobalSaliencyBasedCropping
from Filtering_techniques.Denoise import Denoise
from Filtering_techniques.RandomCropFiltering import RandomCropFiltering
from functions.visualizationFunctions import convert_to_rgb
from functions.loadDatasetFunctions import extract_single_event

def visualize_all_techniques(results, raw_event_data, sample_idx, save_dir="ev_demo_results", scale_factor=3):
    """
    Create a comprehensive visualization with event map on left and filtering techniques on right.
    
    Args:
        results: Dictionary containing results from all filtering techniques
        raw_event_data: Raw event data (dict with x, y, t, p keys) before filtering
        sample_idx: Index of the current sample
        save_dir: Directory to save figures
        scale_factor: Scale factor used in filtering
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure with custom GridSpec layout
    # Left: Event map (2x2 space), Right: 8 techniques (2x4 space)
    fig = plt.figure(figsize=(20, 8))
    gs = GridSpec(2, 6, figure=fig, width_ratios=[1, 1, 1, 1, 1, 1], hspace=0.01, wspace=0.05)
    
    # Event map on the left (spans 2 rows and 2 columns)
    ax_event = fig.add_subplot(gs[:, 0:2])
    
    # 8 technique axes on the right (2 rows, 4 columns)
    ax_techniques = [
        fig.add_subplot(gs[0, 2]),  # Row 0
        fig.add_subplot(gs[0, 3]),
        fig.add_subplot(gs[0, 4]),
        fig.add_subplot(gs[0, 5]),
        fig.add_subplot(gs[1, 2]),  # Row 1
        fig.add_subplot(gs[1, 3]),
        fig.add_subplot(gs[1, 4]),
        fig.add_subplot(gs[1, 5]),
    ]
    
    technique_names = [
        'OMS Filtering',
        'Attention Filtering', 
        'Adaptive Elbow',
        'Goal Oriented',
        'Mean-Std Dev',
        'Global Saliency',
        'Denoise',
        'Random Crop'
    ]
    
    # Create and display event map
    raw_display = None
    if raw_event_data is not None:
        # Create event accumulation map
        raw_display = np.zeros((128, 128), dtype=np.float32)
        
        if isinstance(raw_event_data, dict):
            if 'x' in raw_event_data and 'y' in raw_event_data:
                xs = raw_event_data['x'] if isinstance(raw_event_data['x'], np.ndarray) else np.array(raw_event_data['x'])
                ys = raw_event_data['y'] if isinstance(raw_event_data['y'], np.ndarray) else np.array(raw_event_data['y'])
                # Accumulate events at each location
                np.add.at(raw_display, (ys.astype(int), xs.astype(int)), 1)
        
        # Normalize to 0-255
        raw_display = cv2.normalize(raw_display, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        if raw_display is not None:
            raw_rgb = convert_to_rgb(raw_display)
            ax_event.imshow(raw_rgb)
        else:
            ax_event.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_event.transAxes)
    
    ax_event.set_title('Event Map', fontsize=14, fontweight='bold')
    ax_event.axis('off')
    
    # Plot results from each technique
    for ax, technique_name in zip(ax_techniques, technique_names):
        if technique_name in results and results[technique_name] is not None:
            map_data = results[technique_name]
            map_display = None
            
            # Handle different data types
            if isinstance(map_data, dict):
                # Event dictionary with keys like 'x', 'y', 't', 'p'
                map_display = np.zeros((128, 128), dtype=np.uint8)
                if 'x' in map_data and 'y' in map_data:
                    xs = map_data['x'] if isinstance(map_data['x'], np.ndarray) else np.array(map_data['x'])
                    ys = map_data['y'] if isinstance(map_data['y'], np.ndarray) else np.array(map_data['y'])
                    for x, y in zip(xs, ys):
                        if 0 <= int(x) < 128 and 0 <= int(y) < 128:
                            map_display[int(y), int(x)] = min(255, map_display[int(y), int(x)] + 50)
            elif isinstance(map_data, (list, tuple)):
                # List/tuple of events (x, y, t, p tuples)
                map_display = np.zeros((128, 128), dtype=np.uint8)
                try:
                    for event in map_data:
                        if len(event) >= 2:
                            x, y = event[0], event[1]
                            if 0 <= int(x) < 128 and 0 <= int(y) < 128:
                                map_display[int(y), int(x)] = min(255, map_display[int(y), int(x)] + 50)
                except (TypeError, ValueError):
                    # If iteration fails, try to treat as structured array
                    try:
                        map_display[map_data['y'].astype(int), map_data['x'].astype(int)] = 255
                    except:
                        pass
            elif isinstance(map_data, np.ndarray):
                # Numpy array - normalize and resize if necessary
                map_display = map_data.copy()
                if map_display.dtype != np.uint8:
                    map_display = cv2.normalize(map_display, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                # Resize to 128x128 if smaller (e.g., for cropped images)
                if map_display.shape != (128, 128):
                    map_display = cv2.resize(map_display, (128, 128), interpolation=cv2.INTER_LINEAR)
            
            if map_display is not None:
                # Convert to RGB for display
                map_rgb = convert_to_rgb(map_display)
                ax.imshow(map_rgb)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title(technique_name, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Save figure
    file_path = os.path.join(save_dir, f"sample_{sample_idx:03d}.png")
    fig.savefig(file_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return file_path


if __name__ == "__main__":

    print("DVSGesture Filtering Techniques Demo - First 100 Samples")
    
    # Load DVSGesture dataset
    print("\nLoading DVSGesture training dataset...")
    train_dataset_raw = tonic.datasets.dvsgesture.DVSGesture(save_to="../Datasets", train=True)
    
    scale_factor = 3
    
    # Parameters for filtering techniques
    threshold_OMS = 0.2
    keep_percent = 30
    k_sigma = 0.75
    Use_percentile = True
    percentile = 85
    threshold = 0.15
    sensor_size = (128, 128)
    crop_size = (64, 64)
    
    # Process first 100 samples
    num_samples = min(100, len(train_dataset_raw))
    print(f"Processing first {num_samples} samples...\n")
    
    for sample_idx in range(50, num_samples):
        event, label = train_dataset_raw[sample_idx]
        
        print(f"Sample {sample_idx + 1}/{num_samples} - Label: {label}")
        
        # Dictionary to store visualization data from each technique
        results = {}
        
        try:
            # --- OMS Filtering ---
            print("  Applying OMS Filtering...")
            OMSfilter = OMSFiltering(event, scale_factor, threshold_OMS)
            OMSMap, filtered_event_OMS, I_filtered, Err_OMS = OMSfilter.OMS_filtering()
            print(f"    OMS ERR: {Err_OMS:.4f}")
            results['OMS Filtering'] = I_filtered
            print(f"    ✓ OMS complete")
            
        except Exception as e:
            print(f"    ✗ OMS Filtering error: {str(e)}")
            results['OMS Filtering'] = None
        
        try:
            # --- Attention Filtering ---
            print("  Applying Attention Filtering...")
            Attentionfilter = AttentionFiltering(event, scale_factor)
            filtered_event_Attention, saliency_map, Err_Attention = Attentionfilter.Attention_filtering()
            print(f"    Attention ERR: {Err_Attention:.4f}")
            results['Attention Filtering'] = saliency_map
            print(f"    ✓ Attention complete")
            
        except Exception as e:
            print(f"    ✗ Attention Filtering error: {str(e)}")
            results['Attention Filtering'] = None
        
        try:
            # --- Adaptive Elbow Thresholding ---
            print("  Applying Adaptive Elbow Thresholding...")
            AdaptiveElbowFilter = AdaptiveElbowOMSFiltering(event, scale_factor, threshold_OMS)
            filtered_events, masked_OMS, OMSMap, Err_adElow = AdaptiveElbowFilter.Albowdaptive_thresholding()
            print(f"    Adaptive Elbow ERR: {Err_adElow:.4f}")
            results['Adaptive Elbow'] = masked_OMS
            print(f"    ✓ Adaptive Elbow complete")
            
        except Exception as e:
            print(f"    ✗ Adaptive Elbow Filtering error: {str(e)}")
            results['Adaptive Elbow'] = None
        
        try:
            # --- Goal Oriented Thresholding ---
            print("  Applying Goal Oriented Thresholding...")
            GoalOrientedFilter = MaskGoalOrientedOMSFiltering(event, scale_factor, threshold_OMS)
            filtered_events, masked_OMS, OMSMap, ERR_goal = GoalOrientedFilter.Goadaptive_thresholding(keep_percent)
            print(f"    Goal Oriented ERR: {ERR_goal:.4f}")
            results['Goal Oriented'] = masked_OMS
            print(f"    ✓ Goal Oriented complete")
            
        except Exception as e:
            print(f"    ✗ Goal Oriented Filtering error: {str(e)}")
            results['Goal Oriented'] = None
        
        try:
            # --- Mean and Standard Deviation Thresholding ---
            print("  Applying Mean and Standard Deviation Thresholding...")
            MeanStdFilter = MaskMeanStandardDeviation(event, scale_factor, threshold_OMS)
            filtered_events, ERR_MStd = MeanStdFilter.Mean_std_thresholding(k_sigma)
            print(f"    Mean-StdDev ERR: {ERR_MStd:.4f}")
            results['Mean-Std Dev'] = filtered_events
            print(f"    ✓ Mean-StdDev complete")
            
        except Exception as e:
            print(f"    ✗ Mean-StdDev Filtering error: {str(e)}")
            results['Mean-Std Dev'] = None
        
        try:
            # --- Global Saliency Based Cropping ---
            print("  Applying Global Saliency Based Cropping...")
            GlobalSaliencyCropper = MaskGlobalSaliencyBasedCropping(event, scale_factor, threshold_OMS)
            filtered_events, OMS_norm, cropped_OMS_map, crop_box, ERR_global = GlobalSaliencyCropper.MaskGlobalSaliency_filtering(
                Use_percentile, percentile, threshold
            )
            print(f"    Global Saliency ERR: {ERR_global:.4f}")
            results['Global Saliency'] = cropped_OMS_map
            print(f"    ✓ Global Saliency complete")
            
        except Exception as e:
            print(f"    ✗ Global Saliency Filtering error: {str(e)}")
            results['Global Saliency'] = None
        
        try:
            # --- Denoising Filtering ---
            print("  Applying Denoising Filtering...")
            DenoiseFiltering = Denoise(event, scale_factor)
            events_denoised, ERR = DenoiseFiltering.Denoise_filtering()
            print(f"    Denoise ERR: {ERR:.4f}")
            results['Denoise'] = events_denoised
            print(f"    ✓ Denoise complete")
            
        except Exception as e:
            print(f"    ✗ Denoise Filtering error: {str(e)}")
            results['Denoise'] = None
        
        try:
            # --- Random Cropping Filtering ---
            print("  Applying Random Cropping Filtering...")
            RandomCrop_filtering = RandomCropFiltering(event, scale_factor, sensor_size, crop_size)
            events_cropped, ERR_crop = RandomCrop_filtering.RandomCrop_filtering(sensor_size, crop_size)
            print(f"    Random Crop ERR: {ERR_crop:.4f}")
            # Convert cropped events to image using crop_size (not sensor_size)
            cropped_image = RandomCrop_filtering.events_to_image(events_cropped, crop_size)
            results['Random Crop'] = cropped_image
            print(f"    ✓ Random Crop complete")
            
        except Exception as e:
            print(f"    ✗ Random Crop Filtering error: {str(e)}")
            results['Random Crop'] = None
        
        # Generate comprehensive visualization for all techniques
        print("  Generating comparison visualization...")
        try:
            # Convert raw event to image format for visualization
            raw_event_image = None
            if isinstance(event, dict):
                raw_event_image = event
            else:
                # If it's a raw tonic event, convert it
                xs, ys, timestamps, pols = extract_single_event(event)
                raw_event_image = {
                    'x': xs,
                    'y': ys,
                    't': timestamps,
                    'p': pols
                }
            
            viz_path = visualize_all_techniques(results, raw_event_image, sample_idx, scale_factor=scale_factor)
            print(f"    ✓ Visualization saved to {viz_path}")
        except Exception as e:
            print(f"    ✗ Visualization error: {str(e)}")
        
        print(f"Sample {sample_idx + 1} complete!\n")
    
    print(f"Demo completed! Processed {num_samples} samples.")
