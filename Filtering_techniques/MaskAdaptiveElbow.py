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


class AdaptiveElbowOMSFiltering:
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
        
        # Adaptive Elbow Method for Thresholding
    def adaptive_elbow_threshold(self):
        flat = self.OMS_map.flatten()
        flat = flat[flat > 0]  # remove pure zeros (common in event maps)
        
        if len(flat) < 10:
            return None, 100  # keep everything if too small
        
        flat_sorted = np.sort(flat)
        n = len(flat_sorted)

        # Compute normalized cumulative curve
        cum = np.cumsum(flat_sorted) / flat_sorted.sum()
        x = np.linspace(0, 1, n)

        # Compute curvature: elbow = max distance from diagonal
        diagonal = x
        distances = np.abs(cum - diagonal)
        elbow_idx = np.argmax(distances)

        # compute keep_percent
        keep_percent = 100 * (1 - elbow_idx / n)

        # threshold value
        threshold_value = flat_sorted[elbow_idx]

        print(f"[AUTO] Elbow at index {elbow_idx}, keep ~{keep_percent:.2f}%")
        print(f"[AUTO] Threshold value: {threshold_value:.5f}")

        return threshold_value, keep_percent

    def Albowdaptive_thresholding(self):

        threshold_value, keep_percent = self.adaptive_elbow_threshold()

        # if only zeros and threshold is none
        if threshold_value is None:
            print("[AUTO] Too few OMS values → no filtering applied")
            masked_OMS = self.OMS_map.copy()
            ERR = 0.0
            events_dict = tuple_events_to_event_dict(
                list(zip(self.xs, self.ys, self.timestamps, self.pols))
            )
            return events_dict, masked_OMS, self.OMS_map, ERR

        # build binary mask
        mask = self.OMS_map >= threshold_value
        masked_OMS = self.OMS_map * mask

        filtered_events = []
        max_x_mask, max_y_mask = masked_OMS.shape

        for x, y, t, p in zip(self.xs, self.ys, self.timestamps, self.pols):
            if 0 <= x < max_x_mask and 0 <= y < max_y_mask:
                if mask[x, y]:
                    filtered_events.append((x, y, t, p))

        ERR = 1.0 - (len(filtered_events) / len(self.xs))

        print(f"Filtered events: {len(filtered_events)} (out of {len(self.xs)})")
        print(f"Filtering ERR: {ERR:.4f}")

        events_dict = tuple_events_to_event_dict(filtered_events)

        return events_dict, masked_OMS, self.OMS_map, ERR


    def AdaptiveElbow_filtering_visualization(self, OMS_map, masked_OMS):
        
        print("OMS map stats:", OMS_map.min(), OMS_map.max(), OMS_map.mean())

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
        cv2.putText(background, 'Masked OMS Adptive Elbow', (scaled_width*2+30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,255,0), 2)
        
        cv2.imshow("Visualization", background)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

