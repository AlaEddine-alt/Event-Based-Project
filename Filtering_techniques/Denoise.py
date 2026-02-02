import cv2
import numpy as np
import tonic
import matplotlib.pyplot as plt

# =====================================================
# DENOISE FUNCTION
# =====================================================
def denoise_numpy(events, filter_time=10000):
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

# =====================================================
# LOAD DVSGESTURE DATA
# =====================================================
dataset = tonic.datasets.DVSGesture(save_to="../Datasets", train=False)
events, label = dataset[0]  # choose any sample

# =====================================================
# ORIGINAL EVENT COUNT
# =====================================================
N_original = len(events)

# =====================================================
# APPLY DENOISE
# =====================================================
events_denoised = denoise_numpy(events)
N_denoised = len(events_denoised)
ERR_denoise = 1.0 - (N_denoised / N_original)

print("\n🧹 DVSGesture Denoising")
print(f"Original events = {N_original}")
print(f"Denoised events = {N_denoised}")
print(f"ERR = {ERR_denoise:.4f}")

# =====================================================
# BUILD EVENT MAPS
# =====================================================
def build_event_map(evts):
    xs = evts["x"]
    ys = evts["y"]
    max_x = int(xs.max()) + 1
    max_y = int(ys.max()) + 1
    window_pos = np.zeros((max_y, max_x), dtype=np.uint16)
    for x, y in zip(xs, ys):
        window_pos[y, x] += 1
    return window_pos

window_original = build_event_map(events)
window_denoised = build_event_map(events_denoised)

# =====================================================
# VISUALIZATION
# =====================================================
scale_factor = 4  # scale up for clarity
h, w = window_original.shape
h_scaled, w_scaled = int(h * scale_factor), int(w * scale_factor)

original_resized = cv2.resize(window_original, (w_scaled, h_scaled))
denoised_resized = cv2.resize(window_denoised, (w_scaled, h_scaled))

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