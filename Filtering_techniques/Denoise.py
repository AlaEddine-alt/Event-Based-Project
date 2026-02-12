import cv2
import numpy as np
import tonic
import matplotlib.pyplot as plt

from functions.loadDatasetFunctions import extract_single_event
from functions.adaptFilteredData import structured_to_event_dict

class Denoise:
    def __init__(self, event, scale_factor):
        xs, ys, timestamps, pols = extract_single_event(event)
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
        # Ensure chronological order
        self.events.sort(order="t")
        self.scale_factor = scale_factor
        self.N_original = len(self.events)
        self.xs = xs
        self.ys = ys
        self.timestamps = timestamps
        self.pols = pols
        self.window_original = self.build_event_map(self.events)

    def build_event_map(self, evts):
        xs_filtered = evts['x']
        ys_filtered = evts['y']
        max_x = int(xs_filtered.max()) + 1
        max_y = int(ys_filtered.max()) + 1
        window_pos = np.zeros((max_y, max_x), dtype=np.uint16)
        for x, y in zip(xs_filtered, ys_filtered):
            window_pos[y, x] += 1
        return window_pos

    # DENOISE FUNCTION from Tonic github
    def denoise_numpy(self, filter_time=10000):
        """Drops events that are 'not sufficiently connected to other events in the recording.' In
        practise that means that an event is dropped if no other event occured within a spatial
        neighbourhood of 1 pixel and a temporal neighbourhood of filter_time time units. Useful to
        filter noisy recordings where events occur isolated in time.

        Parameters:
            events: ndarray of shape [num_events, num_event_channels]
            filter_time: maximum temporal distance to next event, otherwise dropped.
                        Lower values will mean higher constraints, therefore less events.

        Returns:
            filtered set of events.
        """
        events = self.events
        assert "x" and "y" and "t" in events.dtype.names

        events_copy = np.zeros_like(events)
        copy_index = 0
        width = int(events["x"].max()) + 1
        height = int(events["y"].max()) + 1
        timestamp_memory = np.zeros((width, height)) + filter_time

        for event in events:
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

        print("\nDVSGesture Denoising")
        print(f"Original events = {self.N_original}")
        print(f"Denoised events = {N_denoised}")
        print(f"ERR = {ERR_denoise:.4f}")
        
        self.window_denoised = self.build_event_map(events_denoised)

        # Convert to classifier-compatible format
        events_dict = structured_to_event_dict(events_denoised)

        return events_dict, ERR_denoise

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

