import cv2
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import tonic

# Assuming initialize_oms and egomotion are defined in functions.OMS_helpers
from functions.OMS_helpers import * 
from functions.visualizationFunctions import draw_graph_with_dots, convert_to_rgb
from functions.loadDatasetFunctions import extract_single_event, reset_windows
from Filtering_techniques.OMSSaliencyMapFiltering import OMSFiltering
from functions.adaptFilteredData import tuple_events_to_event_dict

# ---------------------------
# Config
# ---------------------------
class Config:
    RESOLUTION = [128, 128]  # Resolution of the DVS sensor
    MAX_X = RESOLUTION[0]
    MAX_Y = RESOLUTION[1]
    DROP_RATE = 0  # Percentage of events to drop
    UPDATE_INTERVAL = 0.001  # seconds
    DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    OMS_PARAMS = {
        'size_krn_center': 8,
        'sigma_center': 1,
        'size_krn_surround': 8,
        'sigma_surround': 4,
        'threshold': 0.3, # This is the internal OMS threshold, not the mask threshold
        'tau_memOMS': 0.1,
        'sc': 1,
        'ss': 1
    }

    ATTENTION_PARAMS = {
        'VM_radius': 8, 'VM_radius_group': 15,
        'num_ori': 4, 'b_inh': 3, 'g_inh': 1.0,
        'w_sum': 0.5, 'vm_w': 0.2, 'vm_w2': 0.4,
        'vm_w_group': 0.2, 'vm_w2_group': 0.4,
        'random_init': False, 'lif_tau': 0.3
    }

# ----- Mean and Standard Deviation Thresholding -----

class MaskMeanStandardDeviation:

    def __init__(self, event, scale_factor):
        xs, ys, timestamps, pols = extract_single_event(event)
        window_pos, window_neg, max_x, max_y, numevs = reset_windows(xs, ys, pols)
        self.xs = xs
        self.ys = ys    
        self.timestamps = timestamps
        self.pols = pols
        self.scale_factor = scale_factor
        self.window_pos = window_pos
        self.window_neg = window_neg
        self.max_x = max_x
        self.max_y = max_y
        self.numevs = numevs
        self.events_list = [numevs[0]]
        self.suppressed_list = [numevs[0]]
        self.dropped_list = [0]
        
        self.config = Config()

        # OMS & Attention Initialization

        OMS_filter = OMSFiltering(event, scale_factor)
        self.OMS_map, _, _, _ = OMS_filter.OMS_filtering()

    def Mean_std_thresholding(self, k_sigma):
        # Flatten the OMS map to calculate statistics
        oms_flat = self.OMS_map.flatten()
        
        # Calculate Mean (mu) and Standard Deviation (sigma) of the noise cluster
        mu_noise = np.mean(oms_flat)
        sigma_noise = np.std(oms_flat)
        
        # Calculate the statistical threshold (Theta_mask = mu + k * sigma)
        threshold_value = mu_noise + k_sigma * sigma_noise
        
        # print(f"OMS Stats: Mean={mu_noise:.4f}, StdDev={sigma_noise:.4f}")
        # print(f"Calculated Mask Threshold (k={k_sigma}): {threshold_value:.4f}")
        
        # Create Binary Mask (M): 1=Salient, 0=Suppress
        M = (self.OMS_map > threshold_value).astype(np.uint8)
        
        # Apply Mask to RAW Event Stream
        
        # Retrieve the actual dimensions of the mask for robust indexing
        mask_height, mask_width = M.shape

        # Filter the RAW event stream to ensure all coordinates are within M's exact bounds
        # This prevents IndexErrors if xs_raw/ys_raw contain corrupted or out-of-bounds indices.
        valid_indices = np.logical_and(
            np.logical_and(self.ys < mask_height, self.xs < mask_width), 
            np.logical_and(self.ys >= 0, self.xs >= 0)
        )
        
        # Temporarily subset the raw arrays to only include valid indices
        xs_valid = self.xs[valid_indices]
        ys_valid = self.ys[valid_indices]
        timestamps_valid = self.timestamps[valid_indices]
        pols_valid = self.pols[valid_indices]

        # 1. Get the mask values corresponding to every VALID event's location
        # This is the line that previously failed, but now uses clean indices.
        mask_values_at_events = M[ys_valid, xs_valid]
        
        # 2. Find indices among the VALID events where the mask is TRUE (salient)
        salient_indices_valid = np.where(mask_values_at_events == 1)
        
        # 3. Apply the filtering to generate the new, cleaned event lists
        xs_filtered = xs_valid[salient_indices_valid]
        ys_filtered = ys_valid[salient_indices_valid]
        timestamps_filtered = timestamps_valid[salient_indices_valid]
        pols_filtered = pols_valid[salient_indices_valid]

        filtered_events = list(zip(xs_filtered, ys_filtered, timestamps_filtered, pols_filtered))
        
        # ---------------------------
        # Step 4.5: Event Reduction Ratio (ERR)
        # ---------------------------

        N_original = len(self.xs)
        N_filtered = len(xs_filtered)

        ERR = 1.0 - (N_filtered / N_original)

        # print("========== Event Reduction Stats ==========")
        # print(f"Original events  : {N_original}")
        # print(f"Filtered events  : {N_filtered}")
        # print(f"Suppressed events: {N_original - N_filtered}")
        # print(f"ERR              : {ERR:.4f} ({ERR*100:.2f}%)")
        # print("==========================================")

                
        # Calculate stats for the graph
        self.events_list = [len(self.xs)]
        self.suppressed_list = [len(self.xs) - len(xs_filtered)] 
        self.dropped_list = [0]

        event_dict = tuple_events_to_event_dict(filtered_events)

        return event_dict, ERR

    def MeanStd_filtering_visualization(self, filtered_events, k_sigma):

        # Accumulate Filtered Events for Visualization
        
        window_pos_filtered = np.zeros((self.max_y, self.max_x), dtype=np.uint16)
        for x, y, t, p in filtered_events:
            if y < self.max_y and x < self.max_x and p == 1:
                window_pos_filtered[y, x] += 1

        scaled_height = int(self.max_y * self.scale_factor)
        scaled_width = int(self.max_x * self.scale_factor)

        # visualization 
        background = np.ones((scaled_height, scaled_width*3, 3), dtype=np.uint8) * 255
        
        window_pos_raw_resized = convert_to_rgb(cv2.resize(window_pos_filtered, (scaled_width, scaled_height)))
        OMS_resized = convert_to_rgb(cv2.resize(self.OMS_map, (scaled_width, scaled_height)))
        filtered_map_resized = convert_to_rgb(cv2.resize(window_pos_filtered, (scaled_width, scaled_height)))

        background[:, :scaled_width] = window_pos_raw_resized
        background[:, scaled_width:scaled_width*2] = OMS_resized
        background[:, scaled_width*2:scaled_width*3] = filtered_map_resized
        

        cv2.putText(background, '1. RAW Event Map', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.putText(background, '2. OMS Saliency Map', (scaled_width+30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.putText(background, f'3. Filtered Map (k={k_sigma})', (scaled_width*2+30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        cv2.imshow("Saliency-Based Event Filtering Pipeline", background)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
