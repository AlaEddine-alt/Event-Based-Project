import cv2
import numpy as np
import torch
import time
import tonic
import matplotlib.pyplot as plt

from functions.OMS_helpers import *
from functions.visualizationFunctions import draw_graph_with_dots, convert_to_rgb
from functions.loadDatasetFunctions import extract_single_event, reset_windows
from Filtering_techniques.OMSSaliencyMapFiltering import OMSFiltering
from functions.adaptFilteredData import tuple_events_to_event_dict

# ----- Goal-Oriented OMS Thresholding -----

class MaskGoalOrientedOMSFiltering:

    def __init__(self, event, scale_factor, threshold_OMS):
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
        self.threshold_OMS = threshold_OMS

        # OMS & Attention Initialization

        OMS_filter = OMSFiltering(event, scale_factor, threshold_OMS)
        self.OMS_map, _, _, _ = OMS_filter.OMS_filtering()

    # Mask - goal oriented thresholding
    def goal_oriented_thresholding(self, keep_percent):
        """
        keep_percent: percentage of highest OMS values to keep (0-100)
        """
        # Flatten and compute percentile value
        thr = np.percentile(self.OMS_map, 100 - keep_percent)

        # Create mask: True for pixels we keep
        mask = self.OMS_map >= thr

        # print(f"[Thresholding] Keep top {keep_percent}% → threshold = {thr:.4f}")
        # print(f"[Thresholding] Pixels kept: {mask.sum()}  /  {mask.size}")

        return mask, thr

    def Goadaptive_thresholding(self, keep_percent):

        mask, thr_value = self.goal_oriented_thresholding(keep_percent)

        # Optional: apply mask to visualize
        masked_OMS = self.OMS_map * mask

        filtered_events = []
        max_x_mask, max_y_mask = masked_OMS.shape
        
        for x, y, t, p in zip(self.xs, self.ys, self.timestamps, self.pols):
            if 0 <= x < max_x_mask and 0 <= y < max_y_mask:
                if masked_OMS[y, x] > 0:
                    filtered_events.append((x, y, t, p))

        # print(f"Filtered events: {len(filtered_events)} (out of {len(self.xs)})")
        ERR = 1.0 - (len(filtered_events) / len(self.xs))
        # print(f"Filtering Error Rate (ERR): {ERR:.4f}")

        event_dict = tuple_events_to_event_dict(filtered_events)
        
        return event_dict, masked_OMS, self.OMS_map, ERR
    
    def GoalOriented_filtering_visualization(self, OMS_map, masked_OMS):
        
        # print("OMS map stats:", OMS_map.min(), OMS_map.max(), OMS_map.mean())

        plt.figure(figsize=(6,4))
        plt.hist(OMS_map.flatten(), bins=100, color='gray')
        plt.title("OMS Map Value Distribution")
        plt.xlabel("OMS intensity")
        plt.ylabel("Number of pixels")
        plt.show()

        # Scale components
        scaled_height = int(self.max_y * self.scale_factor)
        scaled_width = int(self.max_x * self.scale_factor)

        background = np.ones((scaled_height, scaled_width*3, 3), dtype=np.uint8) * 255
        window_pos_resized = convert_to_rgb(cv2.resize(self.window_pos, (scaled_width, scaled_height)))
        OMS_resized = convert_to_rgb(cv2.resize(OMS_map, (scaled_width, scaled_height)))
        Mask_resized = convert_to_rgb(cv2.resize(masked_OMS, (scaled_width, scaled_height)))

        background[:, :scaled_width] = window_pos_resized
        background[:, scaled_width:scaled_width*2] = OMS_resized
        background[:, scaled_width*2:] = Mask_resized

        cv2.putText(background, 'Event map', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,255,0), 2)
        cv2.putText(background, 'OMS map', (scaled_width+30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,255,0), 2)
        cv2.putText(background, 'Masked OMS Goal Oriented', (scaled_width*2+30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,255,0), 2)
        
        cv2.imshow("Visualization", background)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
