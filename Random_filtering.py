import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

# ---------------------------
# Config
# ---------------------------
class Config:
    RESOLUTION = [128, 128]
    MAX_X = RESOLUTION[0]
    MAX_Y = RESOLUTION[1]
    DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    RANDOM_KEEP_PROB = 0.3  # Keep 30% of events randomly

config = Config()

# ---------------------------
# Load Events (DVSGesture)
# ---------------------------
import tonic
dataset = tonic.datasets.DVSGesture(save_to="../Datasets", train=False)
events, label = dataset[0]

xs = events["x"].astype(int)
ys = events["y"].astype(int)
pols = events["p"].astype(int)

max_x = int(xs.max()) + 1
max_y = int(ys.max()) + 1

# ---------------------------
# Accumulate RAW events
# ---------------------------
window_pos_raw = np.zeros((max_y, max_x), dtype=np.uint16)
for x, y, p in zip(xs, ys, pols):
    if p == 1:
        window_pos_raw[y, x] += 1

# ---------------------------
# RANDOM FILTERING
# ---------------------------

# Generate random mask per event
random_mask = np.random.rand(len(xs)) < config.RANDOM_KEEP_PROB

xs_rand = xs[random_mask]
ys_rand = ys[random_mask]
pols_rand = pols[random_mask]

# Accumulate filtered events
window_pos_random = np.zeros((max_y, max_x), dtype=np.uint16)
for x, y, p in zip(xs_rand, ys_rand, pols_rand):
    if p == 1:
        window_pos_random[y, x] += 1

# ---------------------------
# Stats for visualization
# ---------------------------
events_list = [len(xs)]
suppressed_list = [len(xs) - len(xs_rand)]
dropped_list = [0]

# ---------------------------
# Visualization helpers
# ---------------------------
def convert_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image

def draw_graph_with_dots(events, suppressed, dropped, width=640, height=480):
    graph_img = np.ones((height, width, 3), dtype=np.uint8) * 255

    max_val = max(events[0], suppressed[0], 1)
    margin = 50
    scale_y = (height - 2 * margin) / max_val

    # axes
    cv2.line(graph_img, (margin, height - margin), (width - margin, height - margin), (0,0,0), 2)
    cv2.line(graph_img, (margin, height - margin), (margin, margin), (0,0,0), 2)

    x = width // 2
    y_raw = height - margin - int(events[0] * scale_y)
    y_sup = height - margin - int(suppressed[0] * scale_y)

    cv2.circle(graph_img, (x, y_raw), 6, (255, 0, 0), -1)
    cv2.circle(graph_img, (x, y_sup), 6, (0, 0, 255), -1)

    cv2.putText(graph_img, f"Raw: {events[0]}", (x-80, y_raw-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
    cv2.putText(graph_img, f"Suppressed: {suppressed[0]}", (x-80, y_sup+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

    cv2.putText(graph_img, "Random Filtering Stats",
                (margin, margin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

    return graph_img

# ---------------------------
# FINAL VISUALIZATION (Same Layout)
# ---------------------------
scale_factor = 4
scaled_h = int(max_y * scale_factor)
scaled_w = int(max_x * scale_factor)

background = np.ones((scaled_h, scaled_w * 3, 3), dtype=np.uint8) * 255

raw_resized = convert_to_rgb(cv2.resize(window_pos_raw, (scaled_w, scaled_h)))
rand_resized = convert_to_rgb(cv2.resize(window_pos_random, (scaled_w, scaled_h)))
graph_resized = cv2.resize(draw_graph_with_dots(events_list, suppressed_list, dropped_list),
                           (scaled_w, scaled_h))

background[:, :scaled_w] = raw_resized
background[:, scaled_w:scaled_w*2] = rand_resized
background[:, scaled_w*2:] = graph_resized

cv2.putText(background, "1. RAW Event Map", (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
cv2.putText(background, "2. Randomly Filtered Events", (scaled_w+30, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
cv2.putText(background, f"3. Event Stats (p={config.RANDOM_KEEP_PROB})",
            (scaled_w*2+30, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

cv2.imshow("Random Event Filtering Baseline", background)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("✅ Random filtering complete")
print("Kept events:", len(xs_rand))
print("Dropped events:", len(xs) - len(xs_rand))
print("Label:", label)

