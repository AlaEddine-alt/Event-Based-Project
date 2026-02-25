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


class MaskGlobalSaliencyBasedCropping():
    
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

    def MaskGlobalSaliency_filtering(self, use_percentile=True, percentile=90, threshold=0.4):

        # Normalize the OMS saliency map
        OMS_norm = (self.OMS_map - self.OMS_map.min()) / (self.OMS_map.max() - self.OMS_map.min() + 1e-6)

        # Choose threshold strategy
        # Option A: fixed threshold (simple)
        # Option B: percentile-based (adaptive)

        if use_percentile:
            thr = np.percentile(OMS_norm, percentile)
            # print(f"Using percentile threshold = {thr:.4f}")
        else:
            thr = threshold  # fixed threshold
            # print(f"Using fixed threshold = {thr}")

        # Create binary mask of salient pixels
        saliency_mask = OMS_norm >= thr

        # print(f"Salient area: {np.sum(saliency_mask)} pixels "
            #f"(out of {saliency_mask.size}, "
            #f"{100*np.mean(saliency_mask):.2f}% of the image)")

        # Step 4: Compute bounding box of salient region

        ys_sal, xs_sal = np.where(saliency_mask)

        if len(xs_sal) == 0:
            print("WARNING: No salient pixels found. Try lowering threshold.")
            x_min = y_min = 0
            x_max = self.max_x
            y_max = self.max_y
        else:
            x_min, x_max = xs_sal.min(), xs_sal.max()
            y_min, y_max = ys_sal.min(), ys_sal.max()

        # print("Proposed cropping coordinates:")
        # print(f"  x_min={x_min}, x_max={x_max}")
        # print(f"  y_min={y_min}, y_max={y_max}")
        # print(f"  Crop size = {(y_max-y_min)} x {(x_max-x_min)}")

        # Step 4.5: Event Reduction Ratio (ERR) — Crop-based

        # Total number of original events
        N_original = len(self.xs)

        # Count events that fall INSIDE the crop
        inside_crop = np.logical_and.reduce((
            self.xs >= x_min,
            self.xs <= x_max,
            self.ys >= y_min,
            self.ys <= y_max
        ))

        N_filtered = np.sum(inside_crop)
        N_suppressed = N_original - N_filtered

        ERR = 1.0 - (N_filtered / N_original)

        # print("\n========== Event Reduction Stats (Crop-based) ==========")
        # print(f"Original events   : {N_original}")
        # print(f"Events in crop    : {N_filtered}")
        # print(f"Suppressed events : {N_suppressed}")
        # print(f"ERR               : {ERR:.4f}")
        # print("=======================================================")

        # Step 5: Crop original event image & OMS map
        cropped_event_map = self.window_pos[y_min:y_max+1, x_min:x_max+1]
        cropped_OMS_map = OMS_norm[y_min:y_max+1, x_min:x_max+1]

        filtered_events = [
        (x, y, t, p)
        for x, y, t, p, keep in zip(
            self.xs, self.ys, self.timestamps, self.pols, inside_crop
        )
        if keep
        ]

        crop_box = {
            "x_min": int(x_min),
            "x_max": int(x_max),
            "y_min": int(y_min),
            "y_max": int(y_max)
        }

        # print("Saved crop box:", crop_box)

        event_dict = tuple_events_to_event_dict(filtered_events)

        return event_dict, OMS_norm, cropped_OMS_map, crop_box, ERR
    
    def MaskGlobalSaliency_filtering_visualization(self, cropped_OMS_map, OMS_norm):

        plt.figure(figsize=(14, 6))

        plt.subplot(1, 3, 1)
        plt.title("Original Event Map")
        plt.imshow(self.window_pos, cmap="gray")
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.title("OMS Saliency Map (Normalized)")
        plt.imshow(OMS_norm, cmap="jet")
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.title("Cropped Salient Region")
        plt.imshow(cropped_OMS_map, cmap="jet")
        plt.colorbar()

        plt.tight_layout()
        plt.show()

        # print("OMS map stats:", self.OMS_map.min(), self.OMS_map.max(), self.OMS_map.mean())

        plt.figure(figsize=(6,4))
        plt.hist(self.OMS_map.flatten(), bins=100, color='gray')
        plt.title("OMS Map Value Distribution")
        plt.xlabel("OMS intensity")
        plt.ylabel("Number of pixels")
        plt.show()


    