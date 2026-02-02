import numpy as np
import tonic
import matplotlib.pyplot as plt

# ===============================
# LOAD DVSGesture
# ===============================
dataset = tonic.datasets.DVSGesture(save_to="../Datasets", train=True)
events, label = dataset[0]

xs = events["x"].astype(int)
ys = events["y"].astype(int)

# ===============================
# ORIGINAL EVENT COUNT
# ===============================
N_original = len(events)

# ===============================
# BUILD FULL EVENT MAP
# ===============================
full_w = xs.max() + 1
full_h = ys.max() + 1

full_map = np.zeros((full_h, full_w), dtype=np.uint16)
for x, y in zip(xs, ys):
    full_map[y, x] += 1

# ===============================
# HAND-FOCUSED CROPPING REGION
# ===============================
# X: middle 50%
x_min = full_w // 4
x_max = (3 * full_w) // 4

# Y: lower 50% (hands region)
y_min = full_h // 2
y_max = full_h

# ===============================
# CROP EVENTS
# ===============================
mask = (
    (xs >= x_min) & (xs <= x_max) &
    (ys >= y_min) & (ys <= y_max)
)

events_cropped = events[mask]
N_cropped = len(events_cropped)
ERR = 1.0 - (N_cropped / N_original)

# ===============================
# BUILD CROPPED MAP (NO BLACK)
# ===============================
xs_c = events_cropped["x"] - x_min
ys_c = events_cropped["y"] - y_min

crop_w = xs_c.max() + 1
crop_h = ys_c.max() + 1

crop_map = np.zeros((crop_h, crop_w), dtype=np.uint16)
for x, y in zip(xs_c, ys_c):
    crop_map[y, x] += 1

# ===============================
# VISUALIZATION
# ===============================
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("DVSGesture — Original")
plt.imshow(full_map, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("DVSGesture — Hand-Focused Crop")
plt.imshow(crop_map, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()

# ===============================
# OUTPUT
# ===============================
print("\n✋ DVSGesture — HAND-FOCUSED CROPPING")
print(f"Original events: {N_original}")
print(f"Remaining events: {N_cropped}")
print(f"ERR = {ERR:.4f}")
print("Label:", label)