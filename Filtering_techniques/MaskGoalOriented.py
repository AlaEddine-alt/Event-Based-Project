import cv2
import numpy as np
import torch
import time
import tonic

from functions.OMS_helpers import *
from functions.attention_helpers import AttentionModule

# ---------------------------
# Config
# ---------------------------
class Config:
    RESOLUTION = [128, 128]  # Resolution of the DVS sensor
    MAX_X = RESOLUTION[0]
    MAX_Y = RESOLUTION[1]
    DROP_RATE = 0  # Percentage of events to drop
    UPDATE_INTERVAL = 0.001  # seconds
    DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    OMS_PARAMS = {
        'size_krn_center': 8,
        'sigma_center': 1,
        'size_krn_surround': 8,
        'sigma_surround': 4,
        'threshold': 0.03,
        'tau_memOMS': 0.1,
        'sc': 1,
        'ss': 1
    }

    ATTENTION_PARAMS = {
        'VM_radius': 8,  # (R0)
        'VM_radius_group': 15,
        'num_ori': 4,
        'b_inh': 3,  # (w)
        'g_inh': 1.0,
        'w_sum': 0.5,
        'vm_w': 0.2,  # (rho)
        'vm_w2': 0.4,
        'vm_w_group': 0.2,
        'vm_w2_group': 0.4,
        'random_init': False,
        'lif_tau': 0.3
    }
# ---------------------------
# Load events from file (x, y, t, p)
# ---------------------------

# data = np.loadtxt('middleDb.txt')  # columns: x y t p
# dataset = tonic.datasets.DVSGesture(save_to="../Datasets", train=False)

data = np.loadtxt('Dsec.txt')  # columns: x y t p

xs = data[:, 0].astype(int)
ys = data[:, 1].astype(int)
timestamps = data[:, 2]         
pols = data[:, 3].astype(int)   

"""
dataset = tonic.datasets.DVSGesture(save_to="../Datasets", train=True)
events, target = dataset[0]

xs = events["x"].astype(int)
ys = events["y"].astype(int)
pols = events["p"].astype(int)
timestamps = events["t"]
"""

print(f"Loaded {len(xs)} events.")

# Auto-resize arrays to fit your data
max_x = int(np.max(xs)) + 1
max_y = int(np.max(ys)) + 1

window_pos = np.zeros((max_y, max_x), dtype=np.uint16)
window_neg = np.zeros((max_y, max_x), dtype=np.uint16)

# Fill windows (accumulate events)
for x, y, p in zip(xs, ys, pols):
    if y < max_y and x < max_x:  # safety check
        if p == 1:
            window_pos[y, x] += 1
        else:
            window_neg[y, x] += 1

numevs = [len(xs)]

# ---------------------------
# OMS & Attention Initialization
# ---------------------------
config = Config()

# ---------------------------
# Visualization functions

def compute_OMS(window_pos):
    OMSpos = torch.tensor(window_pos, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)

    OMSpos_map, indexes_pos = egomotion(OMSpos, net_center, net_surround, config.DEVICE, config.MAX_Y, config.MAX_X,
                                        config.OMS_PARAMS['threshold'])

    OMSpos_map = OMSpos_map.squeeze(0).squeeze(0).cpu().detach().numpy()

    print("OMS map stats:", OMSpos_map.min(), OMSpos_map.max(), OMSpos_map.mean())
    
    return OMSpos_map, indexes_pos

def draw_graph_with_dots(events, suppressed, dropped, width=640, height=480):
    graph_img = np.ones((height, width, 3), dtype=np.uint8) * 255

    max_events = max(events + suppressed + dropped, default=1)
    margin = 50
    scale_x = (width - 2 * margin) / len(events) if events else 1
    scale_y = (height - 2 * margin) / max_events

    cv2.line(graph_img, (margin, height - margin), (width - margin, height - margin), (0,0,0), 2)
    cv2.line(graph_img, (margin, height - margin), (margin, margin), (0,0,0), 2)

    for i in range(len(events)):
        x = margin + int(i * scale_x)
        y_events = height - margin - int(events[i]*scale_y)
        cv2.circle(graph_img, (x, y_events), 4, (0,0,255), -1)
        y_suppressed = height - margin - int(suppressed[i]*scale_y)
        cv2.circle(graph_img, (x, y_suppressed), 4, (255,0,0), -1)
    return graph_img

def convert_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image

# Adaptive Elbow Method for Thresholding
def adaptive_elbow_threshold(OMS_map):
    flat = OMS_map.flatten()
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


# Mask - goal oriented thresholding
def goal_oriented_thresholding(OMS_map, keep_percent):
    """
    keep_percent: percentage of highest OMS values to keep (0-100)
    """
    # Flatten and compute percentile value
    thr = np.percentile(OMS_map, 100 - keep_percent)
    # thr, keep_percent = adaptive_elbow_threshold(OMS_map)

    # Create mask: True for pixels we keep
    mask = OMS_map >= thr

    print(f"[Thresholding] Keep top {keep_percent}% → threshold = {thr:.4f}")
    print(f"[Thresholding] Pixels kept: {mask.sum()}  /  {mask.size}")

    return mask, thr


# ---------------------------
# Main loop
# ---------------------------

if __name__ == "__main__":

    net_center, net_surround = initialize_oms(config.DEVICE, config.OMS_PARAMS)
    net_attention = AttentionModule(**config.ATTENTION_PARAMS)

    # scale_factor = 4 # for DVSGesture
    scale_factor = 0.75  # for DSEC
    events_list = [numevs[0]]      # total events
    suppressed_list = [numevs[0]]  # fake OMS indexes sum
    dropped_list = [0]

    OMS_map, indexes = compute_OMS(window_pos)

    # ----- Goal-Oriented OMS Thresholding -----
    # mask, thr_value = goal_oriented_thresholding(OMS_map, 5)

    # ----- Adaptive Elbow Method Thresholding -----
    mask, thr_value = adaptive_elbow_threshold(OMS_map)

    # Optional: apply mask to visualize
    masked_OMS = OMS_map * mask

    filtered_events = []
    max_x_mask, max_y_mask = masked_OMS.shape

    """
    for x, y, t, p in events:
        if x < max_x_mask and y < max_y_mask:
            if masked_OMS[x, y]:
                filtered_events.append((x, y, t, p))
    """

    for x, y, t, p in zip(xs, ys, timestamps, pols):
        if 0 <= x < max_x_mask and 0 <= y < max_y_mask:
            if masked_OMS[x, y] > 0:   # CORRETTO: [y, x]
                filtered_events.append((x, y, t, p))

    print(f"Filtered events: {len(filtered_events)} (out of {len(xs)})")

    # ---------------------------
    # Inspect OMS value distribution
    # ---------------------------
    import matplotlib.pyplot as plt

    print("OMS map stats:", OMS_map.min(), OMS_map.max(), OMS_map.mean())

    plt.figure(figsize=(6,4))
    plt.hist(OMS_map.flatten(), bins=100, color='gray')
    plt.title("OMS Map Value Distribution")
    plt.xlabel("OMS intensity")
    plt.ylabel("Number of pixels")
    plt.show()


    # Scale components
    scaled_height = int(max_y * scale_factor)
    scaled_width = int(max_x * scale_factor)

    background = np.ones((scaled_height, scaled_width*3, 3), dtype=np.uint8) * 255
    window_pos_resized = convert_to_rgb(cv2.resize(window_pos, (scaled_width, scaled_height)))
    OMS_resized = convert_to_rgb(cv2.resize(OMS_map, (scaled_width, scaled_height)))
    Mask_resized = convert_to_rgb(cv2.resize(masked_OMS, (scaled_width, scaled_height)))
    graph_img_resized = cv2.resize(draw_graph_with_dots(events_list, suppressed_list, dropped_list),
                                (scaled_width, scaled_height))

    background[:, :scaled_width] = window_pos_resized
    background[:, scaled_width:scaled_width*2] = OMS_resized
    background[:, scaled_width*2:] = Mask_resized


    cv2.putText(background, 'Event map', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,255,0), 2)
    cv2.putText(background, 'OMS map', (scaled_width+30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,255,0), 2)
    cv2.putText(background, 'Masked OMS', (scaled_width*2+30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,255,0), 2)
    
    cv2.imshow("Visualization", background)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
