import cv2
import numpy as np
import torch
import time
import tonic

from functions.OMS_helpers import *
from functions.attention_helpers import AttentionModule
from functions.visualizationFunctions import draw_graph_with_dots, convert_to_rgb
from functions.loadDatasetFunctions import extract_single_event, reset_windows
from functions.computeOMSFunction import compute_OMS
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
        'size_krn_center': 7,
        'sigma_center': 1,
        'size_krn_surround': 7,
        'sigma_surround': 4,
        'threshold': 0.03,
        'tau_memOMS': 0.1,
        'sc': 1,
        'ss': 1
    }

    ATTENTION_PARAMS = {
        'VM_radius': 8,  # (R0)
        'VM_radius_group': 15,
        'num_ori': 4,
        'b_inh': 3,  # (w)
        'g_inh': 1.0,
        'w_sum': 0.5,
        'vm_w': 0.2,  # (rho)
        'vm_w2': 0.4,
        'vm_w_group': 0.2,
        'vm_w2_group': 0.4,
        'random_init': False,
        'lif_tau': 0.3
    }

# ---------------------------
# Main loop
# ---------------------------

class OMSFiltering:

    def __init__(self, event, scale_factor):

        # OMS & Attention Initialization
        
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


    def OMS_filtering(self):

        net_center, net_surround = initialize_oms(self.config.DEVICE, self.config.OMS_PARAMS)
        net_attention = AttentionModule(**self.config.ATTENTION_PARAMS)


        OMS_map, indexes = compute_OMS(self.window_pos, net_center, net_surround, self.config)

        # --- Saliency Map Retrieval and Normalization ---
        
        # 1. Convert to float32 for normalization and filtering operations
        S = OMS_map.astype(np.float32)
        
        # 2. Normalize Saliency Map (S) to a 0-to-1 range
        min_val = np.min(S)
        max_val = np.max(S)
        # Use a safety check for division by zero (in case the map is uniformly zero)
        if max_val > min_val:
            S_normalized = (S - min_val) / (max_val - min_val)
        else:
            # If no variation, the map provides no guidance, default to no-filtering (1)
            S_normalized = np.ones_like(S)

        # The variable 'S_normalized' is the final filtering mask (S)
        

        # 2. Prepare the Input Image (I)
        I = self.window_pos.astype(np.float32) # The raw event map is our input I
        
        # --- NEW: Ensure all matrices have consistent shape ---
        target_shape = S_normalized.shape # Should be (473, 633)

        # CRITICAL CHECK: Make sure I's shape matches S_normalized's shape
        if I.shape != target_shape:
            print(f"Warning: Input image shape {I.shape} does not match Saliency map shape {target_shape}. Resizing input.")
            # Resize I to match S_normalized shape (This might happen if max_x/max_y was calculated differently)
            I = cv2.resize(I, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
        # --- END CRITICAL CHECK ---

        # 3. Define Filter Extremes (Tune these sigma values)
        sigma_min = 0.5  # Minimal blur, preserves details (for salient objects)
        sigma_max = 5.0  # Aggressive blur, removes noise (for background)

        # Calculate kernel sizes (must be positive odd integers)
        def get_ksize(sigma):
            return int(2 * np.ceil(2 * sigma) + 1)
        
        ksize_min = get_ksize(sigma_min)
        ksize_max = get_ksize(sigma_max)
        
        # 4. Create the two extreme filtered images
        # I_preserved: Lightly filtered (used when S is high)
        I_preserved = cv2.GaussianBlur(I, (ksize_min, ksize_min), sigma_min)

        # I_smooth: Heavily filtered (used when S is low)
        I_smooth = cv2.GaussianBlur(I, (ksize_max, ksize_max), sigma_max)

        # 5. Apply Weighted Blending: I_filtered = I_preserved * S + I_smooth * (1 - S)
        one_minus_S = 1.0 - S_normalized
        I_filtered = I_preserved * S_normalized + I_smooth * one_minus_S
        
        # --- End of Filtering ---
        
        #Convert filtered result to an 8-bit image for clear visualization
        # Normalize filtered map to 0-255 scale
        I_filtered_normalized = I_filtered / np.max(I_filtered)
        I_filtered_8bit = (I_filtered_normalized * 255).astype(np.uint8)


        # ---------------------------
        # Event-level filtering using OMS Saliency
        # ---------------------------

        # Saliency threshold (tune this)
        saliency_threshold = 0.3  

        # Boolean mask of salient pixels
        saliency_mask = S_normalized >= saliency_threshold

        # Lists for filtered / suppressed events
        filtered_xs, filtered_ys, filtered_ts, filtered_ps = [], [], [], []
        suppressed_xs, suppressed_ys = [], []

        for x, y, t, p in zip(self.xs, self.ys, self.timestamps, self.pols):
            if y < saliency_mask.shape[0] and x < saliency_mask.shape[1]:
                if saliency_mask[y, x]:
                    filtered_xs.append(x)
                    filtered_ys.append(y)
                    filtered_ts.append(t)
                    filtered_ps.append(p)
                else:
                    suppressed_xs.append(x)
                    suppressed_ys.append(y)

        filtered_events = list(zip(filtered_xs, filtered_ys, filtered_ts, filtered_ps))

        # Statistics
        num_total_events = len(self.xs)
        num_filtered_events = len(filtered_xs)
        num_suppressed_events = len(suppressed_xs)
        ERR = 1.0 - (num_filtered_events / num_total_events)

        # print("----- OMS Filtering Stats -----")
        # print(f"Total events     : {num_total_events}")
        # print(f"Retained events  : {num_filtered_events}")
        # print(f"Filtered events  : {num_suppressed_events}")
        # print(f"Filtered ratio   : {ERR:.4f}")

        events_dict = tuple_events_to_event_dict(filtered_events)

        return OMS_map, events_dict, I_filtered_8bit, ERR

    def OMS_filtering_visualization(self, OMS_map, I_filtered_8bit):
        # ---------------------------
        # Update Visualization
        # ---------------------------
        
        # Scale components
        scaled_height = int(self.max_y * self.scale_factor)
        scaled_width = int(self.max_x * self.scale_factor)

        background = np.ones((scaled_height, scaled_width*3, 3), dtype=np.uint8) * 255

        window_pos_resized = convert_to_rgb(cv2.resize(self.window_pos, (scaled_width, scaled_height)))
        OMS_resized = convert_to_rgb(cv2.resize(OMS_map, (scaled_width, scaled_height)))
        graph_img_resized = cv2.resize(draw_graph_with_dots(self.events_list, self.suppressed_list, self.dropped_list),
                                    (scaled_width, scaled_height))
        
        # We will now add the filtered map to the visualization
        
        # Resize the filtered image for display
        I_filtered_resized = convert_to_rgb(cv2.resize(I_filtered_8bit, (scaled_width, scaled_height)))
        
        # The 'background' image needs to be wider to fit the 4th image (Event, OMS, Filtered, Graph)
        background = np.ones((scaled_height, scaled_width*4, 3), dtype=np.uint8) * 255

        # Original window_pos
        window_pos_resized = convert_to_rgb(cv2.resize(self.window_pos, (scaled_width, scaled_height)))
        # OMS map
        OMS_resized = convert_to_rgb(cv2.resize(OMS_map, (scaled_width, scaled_height)))
        # Graph
        graph_img_resized = cv2.resize(draw_graph_with_dots(self.events_list, self.suppressed_list, self.dropped_list),
                                    (scaled_width, scaled_height))

    # Place images into the wide background
        background[:, :scaled_width] = window_pos_resized
        background[:, scaled_width:scaled_width*2] = OMS_resized
        background[:, scaled_width*2:scaled_width*3] = I_filtered_resized  # NEW FILTERED MAP
        background[:, scaled_width*3:] = graph_img_resized

        # Update Text Labels
        cv2.putText(background, 'Event map', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,255,0), 2)
        cv2.putText(background, 'OMS map', (scaled_width+30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,255,0), 2)
        cv2.putText(background, 'Filtered map', (scaled_width*2+30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,255,0), 2) # NEW LABEL

        cv2.imshow("Visualization", background)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
