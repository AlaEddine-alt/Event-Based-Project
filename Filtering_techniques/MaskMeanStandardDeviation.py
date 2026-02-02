import cv2
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import tonic

# Assuming initialize_oms and egomotion are defined in functions.OMS_helpers
from functions.OMS_helpers import * 
from functions.attention_helpers import AttentionModule

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

# ---------------------------
# Laod DSEC dataset
# ---------------------------

data = np.loadtxt('Dsec.txt')  # columns: x y t p

xs_raw = data[:, 0].astype(int)
ys_raw = data[:, 1].astype(int)
timestamps_raw = data[:, 2]         
pols_raw = data[:, 3].astype(int)   

# you have to put the scale_factor=0,6 

# ---------------------------
# Load DVS Gesture dataset
# ---------------------------
"""
data = tonic.datasets.DVSGesture(save_to="../Datasets", train=True)
events, targets = data[0]  # Get the first sample
xs_raw = events["x"].astype(int)
ys_raw = events["y"].astype(int)    
pols_raw = events["p"].astype(int)
timestamps_raw = events["t"]

# you have to put the scale_factor=3
"""


# Auto-determine map dimensions based on the absolute maximum index observed
max_x = int(np.max(xs_raw)) + 1
max_y = int(np.max(ys_raw)) + 1

# ---------------------------
# Visualization functions
# ---------------------------

def compute_OMS(window_pos, net_center, net_surround, config):
    OMSpos = torch.tensor(window_pos, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
    OMSpos_map, indexes_pos = egomotion(OMSpos, net_center, net_surround, config.DEVICE, config.MAX_Y, config.MAX_X,
                                        config.OMS_PARAMS['threshold'])
    OMSpos_map = OMSpos_map.squeeze(0).squeeze(0).cpu().detach().numpy()
    return OMSpos_map, indexes_pos

def draw_graph_with_dots(events, suppressed, dropped, width=640, height=480):
    graph_img = np.ones((height, width, 3), dtype=np.uint8) * 255
    # Max events for scaling needs to consider the baseline (total raw events)
    max_raw_events = events[0] if events else 1 
    max_suppressed = suppressed[0] if suppressed else 0
    max_value = max(max_raw_events, max_suppressed, 1)

    margin = 50
    scale_x = (width - 2 * margin) / len(events) if events else 1
    scale_y = (height - 2 * margin) / max_value
    cv2.line(graph_img, (margin, height - margin), (width - margin, height - margin), (0,0,0), 2)
    cv2.line(graph_img, (margin, height - margin), (margin, margin), (0,0,0), 2)
    
    # Assuming we are only plotting one point for total events vs suppressed
    if len(events) > 0:
        # Raw Total (Blue dot)
        x = margin + int(0.5 * scale_x)
        y_events = height - margin - int(events[0] * scale_y)
        cv2.circle(graph_img, (x, y_events), 4, (255, 0, 0), -1) # Blue
        cv2.putText(graph_img, f'Raw: {events[0]}', (x + 10, y_events - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        # Suppressed Total (Red dot)
        y_suppressed = height - margin - int(suppressed[0] * scale_y)
        cv2.circle(graph_img, (x, y_suppressed), 4, (0, 0, 255), -1) # Red
        cv2.putText(graph_img, f'Suppr: {suppressed[0]}', (x + 10, y_suppressed + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        
    cv2.putText(graph_img, 'Event Count Comparison', (margin, margin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    return graph_img

def convert_to_rgb(image):
    # Converts a single-channel image (like an event map) to a 3-channel BGR image.
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) 
    else:
        return image

# ---------------------------
# Main loop
# ---------------------------

if __name__ == "__main__":
    config = Config()

    # --- Step 1: Accumulate RAW Events for OMS Input and Baseline ---
    window_pos_raw = np.zeros((max_y, max_x), dtype=np.uint16) 
    for x, y, p in zip(xs_raw, ys_raw, pols_raw):
        # We must ignore events outside the defined boundaries for accumulation
        if y < max_y and x < max_x and p == 1:
            window_pos_raw[y, x] += 1
            
    # --- Step 2: Initialize OMS kernels and Calculate Saliency Map ---
    net_center, net_surround = initialize_oms(config.DEVICE, config.OMS_PARAMS)
    net_attention = AttentionModule(**config.ATTENTION_PARAMS) 
    
    # OMS_map is the saliency guide (S)
    OMS_map, _ = compute_OMS(window_pos_raw, net_center, net_surround, config) 
    
    # ----------------------------------------------------
    # CRITICAL STEP 3: Statistical Mask Generation (3-Sigma Rule)
    # ----------------------------------------------------
    
    # Flatten the OMS map to calculate statistics
    oms_flat = OMS_map.flatten()
    
    # Calculate Mean (mu) and Standard Deviation (sigma) of the noise cluster
    mu_noise = np.mean(oms_flat)
    sigma_noise = np.std(oms_flat)
    
    # Define the scaling factor (k). 3 is the standard for strong outlier detection.
    k_sigma = 2.0 
    
    # Calculate the statistical threshold (Theta_mask = mu + k * sigma)
    threshold_value = mu_noise + k_sigma * sigma_noise
    
    print(f"OMS Stats: Mean={mu_noise:.4f}, StdDev={sigma_noise:.4f}")
    print(f"Calculated Mask Threshold (k={k_sigma}): {threshold_value:.4f}")
    
    # Create Binary Mask (M): 1=Salient, 0=Suppress
    M = (OMS_map > threshold_value).astype(np.uint8)
    
    # ----------------------------------------------------
    # CRITICAL STEP 4: Apply Mask to RAW Event Stream
    # ----------------------------------------------------
    
    # NEW: Retrieve the actual dimensions of the mask for robust indexing
    mask_height, mask_width = M.shape

    # NEW: Filter the RAW event stream to ensure all coordinates are within M's exact bounds
    # This prevents IndexErrors if xs_raw/ys_raw contain corrupted or out-of-bounds indices.
    valid_indices = np.logical_and(
        np.logical_and(ys_raw < mask_height, xs_raw < mask_width), 
        np.logical_and(ys_raw >= 0, xs_raw >= 0)
    )
    
    # Temporarily subset the raw arrays to only include valid indices
    xs_valid = xs_raw[valid_indices]
    ys_valid = ys_raw[valid_indices]
    timestamps_valid = timestamps_raw[valid_indices]
    pols_valid = pols_raw[valid_indices]

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
    
    # ---------------------------
    # Step 4.5: Event Reduction Ratio (ERR)
    # ---------------------------

    N_original = len(xs_raw)
    N_filtered = len(xs_filtered)

    ERR = 1.0 - (N_filtered / N_original)

    print("========== Event Reduction Stats ==========")
    print(f"Original events  : {N_original}")
    print(f"Filtered events  : {N_filtered}")
    print(f"Suppressed events: {N_original - N_filtered}")
    print(f"ERR              : {ERR:.4f} ({ERR*100:.2f}%)")
    print("==========================================")

    
    # --- Step 5: Accumulate Filtered Events for Visualization ---
    
    window_pos_filtered = np.zeros((max_y, max_x), dtype=np.uint16)
    for x, y, p in zip(xs_filtered, ys_filtered, pols_filtered):
        if y < max_y and x < max_x and p == 1:
            window_pos_filtered[y, x] += 1
            
    # Calculate stats for the graph
    events_list = [len(xs_raw)]
    suppressed_list = [len(xs_raw) - len(xs_filtered)] 
    dropped_list = [0]


    # ---------------------------
    # Step 6: Visualization
    # ---------------------------
    
    scale_factor = 0.6
    scaled_height = int(max_y * scale_factor)
    scaled_width = int(max_x * scale_factor)

    # Setup 4-panel visualization 
    background = np.ones((scaled_height, scaled_width*4, 3), dtype=np.uint8) * 255
    
    window_pos_raw_resized = convert_to_rgb(cv2.resize(window_pos_raw, (scaled_width, scaled_height)))
    OMS_resized = convert_to_rgb(cv2.resize(OMS_map, (scaled_width, scaled_height)))
    filtered_map_resized = convert_to_rgb(cv2.resize(window_pos_filtered, (scaled_width, scaled_height)))
    graph_img_resized = cv2.resize(draw_graph_with_dots(events_list, suppressed_list, dropped_list),
                                (scaled_width, scaled_height))

    background[:, :scaled_width] = window_pos_raw_resized
    background[:, scaled_width:scaled_width*2] = OMS_resized
    background[:, scaled_width*2:scaled_width*3] = filtered_map_resized
    background[:, scaled_width*3:] = graph_img_resized
    

    cv2.putText(background, '1. RAW Event Map', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    cv2.putText(background, '2. OMS Saliency Map', (scaled_width+30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    cv2.putText(background, f'3. Filtered Map (k={k_sigma})', (scaled_width*2+30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    cv2.putText(background, '4. Event Stats', (scaled_width*3+30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)


    cv2.imshow("Saliency-Based Event Filtering Pipeline", background)
    cv2.waitKey(0)
    cv2.destroyAllWindows()