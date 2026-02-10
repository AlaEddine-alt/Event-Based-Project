import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import tonic

from functions.visualizationFunctions import draw_graph_with_dots, convert_to_rgb
from functions.loadDatasetFunctions import load_events, reset_windows

# ---------------------------
# Config
# ---------------------------
class Config:
    RESOLUTION = [128, 128]
    MAX_X = RESOLUTION[0]
    MAX_Y = RESOLUTION[1]
    DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

config = Config()

class RandomEventFiltering:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        xs, ys, timestamps, pols, scale_factor = load_events(dataset_name)
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


    def Random_filtering(self, random_keep_prob):
        
        # Generate random mask per event
        random_mask = np.random.rand(len(self.xs)) < random_keep_prob

        xs_rand = self.xs[random_mask]
        ys_rand = self.ys[random_mask]
        pols_rand = self.pols[random_mask]
        timestamps_rand = self.timestamps[random_mask]

        filtered_events = list(zip(xs_rand, ys_rand, timestamps_rand, pols_rand))

        # Accumulate filtered events
        window_pos_random = np.zeros((self.max_y, self.max_x), dtype=np.uint16)
        for x, y, p in zip(xs_rand, ys_rand, pols_rand):
            if p == 1:
                window_pos_random[y, x] += 1

        self.events_list = [len(self.xs)]
        self.suppressed_list = [len(self.xs) - len(xs_rand)]
        self.dropped_list = [0]

        return window_pos_random, xs_rand, filtered_events
    
    def Random_filtering_visualization(self, window_pos_random, xs_rand):

        scaled_h = int(self.max_y * self.scale_factor)
        scaled_w = int(self.max_x * self.scale_factor)

        background = np.ones((scaled_h, scaled_w * 2, 3), dtype=np.uint8) * 255

        raw_resized = convert_to_rgb(cv2.resize(self.window_pos, (scaled_w, scaled_h)))
        rand_resized = convert_to_rgb(cv2.resize(window_pos_random, (scaled_w, scaled_h)))

        background[:, :scaled_w] = raw_resized
        background[:, scaled_w:scaled_w*2] = rand_resized

        cv2.putText(background, "1. RAW Event Map", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.putText(background, "2. Randomly Filtered Events", (scaled_w+30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        cv2.imshow("Random Event Filtering Baseline", background)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("Random filtering complete")
        print("Kept events:", len(xs_rand))
        print("Dropped events:", len(self.xs) - len(xs_rand))



