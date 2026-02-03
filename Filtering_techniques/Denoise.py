import cv2
import numpy as np
import tonic
import matplotlib.pyplot as plt

from functions.loadDatasetFunctions import load_events

class Denoise:
    def __init__(self, datset_name):
        xs, ys, timestamps, pols, scale_factor = load_events(datset_name)
        self.events = np.zeros(len(xs), dtype=[
            ('x', np.int16),
            ('y', np.int16),
            ('t', np.int64),
            ('p', np.int8),
        ])
        self.events['x'] = xs
        self.events['y'] = ys
        self.events['t'] = timestamps
        self.events['p'] = pols
        self.scale_factor = scale_factor
        self.N_original = len(self.events)
        self.xs = xs
        self.ys = ys
        self.timestamps = timestamps
        self.pols = pols
        self.window_original = self.build_event_map(self.events)

    def build_event_map(self, evts):
        max_x = int(self.xs.max()) + 1
        max_y = int(self.ys.max()) + 1
        window_pos = np.zeros((max_y, max_x), dtype=np.uint16)
        for x, y in zip(self.xs, self.ys):
            window_pos[y, x] += 1
        return window_pos

    # DENOISE FUNCTION from Tonic github
    def denoise_numpy(self, filter_time=10000):
        events_copy = np.zeros_like(self.events)
        copy_index = 0

        width = int(self.events["x"].max()) + 1
        height = int(self.events["y"].max()) + 1

        timestamp_memory = np.zeros((width, height)) + filter_time

        for event in self.events:
            x = int(event["x"])
            y = int(event["y"])
            t = event["t"]

            timestamp_memory[x, y] = t + filter_time

            if (
                (x > 0 and timestamp_memory[x - 1, y] > t)
                or (x < width - 1 and timestamp_memory[x + 1, y] > t)
                or (y > 0 and timestamp_memory[x, y - 1] > t)
                or (y < height - 1 and timestamp_memory[x, y + 1] > t)
            ):
                events_copy[copy_index] = event
                copy_index += 1

        return events_copy[:copy_index]
    
    def Denoise_filtering(self):
        # APPLY DENOISE
        events_denoised = self.denoise_numpy()
        N_denoised = len(events_denoised)
        ERR_denoise = 1.0 - (N_denoised / self.N_original)

        print("\n🧹 DVSGesture Denoising")
        print(f"Original events = {self.N_original}")
        print(f"Denoised events = {N_denoised}")
        print(f"ERR = {ERR_denoise:.4f}")
        
        self.window_denoised = self.build_event_map(events_denoised)

        return events_denoised

    def Denoise_filtering_visualization(self):

        h, w = self.window_original.shape
        h_scaled, w_scaled = int(h * self.scale_factor), int(w * self.scale_factor)

        original_resized = cv2.resize(self.window_original, (w_scaled, h_scaled))
        denoised_resized = cv2.resize(self.window_denoised, (w_scaled, h_scaled))

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.title("Original Event Map")
        plt.imshow(original_resized, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Denoised Event Map")
        plt.imshow(denoised_resized, cmap="gray")
        plt.axis("off")

        plt.tight_layout()
        plt.show()