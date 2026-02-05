import cv2
import tonic 
import numpy as np
import matplotlib.pyplot as plt

from functions.loadDatasetFunctions import extract_single_event

class RandomCropFiltering:
    def __init__(self, event, scale_factor, sensor_size, crop_size):
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
        self.scale_factor = scale_factor
        self.N_original = len(self.events)
        self.xs = xs
        self.ys = ys
        self.timestamps = timestamps
        self.pols = pols
        self.sensor_size = sensor_size
        self.crop_size = crop_size

    def crop_numpy(self, events, sensor_size, target_size):
        """Crops the sensor size to a smaller sensor.

        x' = x - new_sensor_start_x
        y' = y - new_sensor_start_y

        Parameters:
            events: ndarray of shape [num_events, num_event_channels]
            sensor_size: size of the sensor that was used [W,H]
            target_size: size of the sensor that was used [W',H']

        Returns:
            events - events within the crop box
            sensor_size - cropped to target_size
        """

        assert target_size[0] <= sensor_size[0] and target_size[1] <= sensor_size[1]
        assert "x" and "y" in events.dtype.names

        x_start_ind = int(np.random.rand() * (sensor_size[0] - target_size[0]))
        y_start_ind = int(np.random.rand() * (sensor_size[1] - target_size[1]))

        x_end_ind = x_start_ind + target_size[0]
        y_end_ind = y_start_ind + target_size[1]

        event_mask = (
            (events["x"] >= x_start_ind)
            * (events["x"] < x_end_ind)
            * (events["y"] >= y_start_ind)
            * (events["y"] < y_end_ind)
        )

        events = events[event_mask, ...]
        events["x"] -= x_start_ind
        events["y"] -= y_start_ind

        return events

    def RandomCrop_filtering(self, sensor_size, crop_size):
        events_cropped = self.crop_numpy(
            events=self.events,
            sensor_size=sensor_size,
            target_size=crop_size
        )

        N_cropped = len(events_cropped)
        ERR_cropped = 1.0 - (N_cropped / self.N_original)

        print("Random Cropping")
        print(f"Original events = {self.N_original}")
        print(f"Cropped events  = {N_cropped}")
        print(f"ERR            = {ERR_cropped:.4f}")

        return events_cropped, ERR_cropped
    
    def events_to_image(self, events, sensor_size):

        W, H = sensor_size
        img = np.zeros((H, W), dtype=np.float32)

        np.add.at(img, (events["y"], events["x"]), 1)

        return img
    
    def RandomCrop_filtering_visualization(self, cropped_events):
        # Build event-count images
        original_img = self.events_to_image(self.events, self.sensor_size)
        cropped_img = self.events_to_image(cropped_events, self.crop_size)

        # Scale for visualization
        h, w = original_img.shape
        h_scaled, w_scaled = int(h * self.scale_factor), int(w * self.scale_factor)

        original_resized = cv2.resize(original_img, (w_scaled, h_scaled))
        cropped_resized = cv2.resize(cropped_img, (w_scaled, h_scaled))

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.title("Original Event Map")
        plt.imshow(original_resized, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Cropped Event Map")
        plt.imshow(cropped_resized, cmap="gray")
        plt.axis("off")

        plt.tight_layout()
        plt.show()