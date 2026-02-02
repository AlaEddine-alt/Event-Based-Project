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

#data = np.loadtxt('middleDb.txt')  # columns: x y t p



# ---------------------------
# Laod DSEC dataset
# ---------------------------

data = np.loadtxt('Dsec.txt')  # columns: x y t p

xs = data[:, 0].astype(int)
ys = data[:, 1].astype(int)
timestamps = data[:, 2]         
pols = data[:, 3].astype(int)   

# you have to put the scale_factor=0,6 

"""
# ---------------------------
# Load DVS Gesture dataset
# ---------------------------

data = tonic.datasets.DVSGesture(save_to="../Datasets", train=True)
events, targets = data[0]  # Get the first sample

xs = events["x"].astype(int)
ys = events["y"].astype(int)   
pols = events["p"].astype(int)       
timestamps = events["t"]  
# you have to put the scale_factor=3

"""


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

print(f"number of events {len(xs)}")

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

# ---------------------------
# Main loop
# ---------------------------

if __name__ == "__main__":

    net_center, net_surround = initialize_oms(config.DEVICE, config.OMS_PARAMS)
    net_attention = AttentionModule(**config.ATTENTION_PARAMS)

    scale_factor = 0.6
    events_list = [numevs[0]]      # total events
    suppressed_list = [numevs[0]]  # fake OMS indexes sum
    dropped_list = [0]

    OMS_map, indexes = compute_OMS(window_pos)
    
    # =======================================================
    #  1st FILTERING SCENARIO — GLOBAL SALIENCY CROP DISCOVERY
    # =======================================================

    print("\n--- GLOBAL SALIENCY-BASED CROP DISCOVERY ---")

    # ------------------------------------------
    # Step 1: Normalize the OMS saliency map
    # ------------------------------------------
    OMS_norm = (OMS_map - OMS_map.min()) / (OMS_map.max() - OMS_map.min() + 1e-6)

    # ------------------------------------------
    # Step 2: Choose threshold strategy
    #   Option A: fixed threshold (simple)
    #   Option B: percentile-based (adaptive)
    # ------------------------------------------

    USE_PERCENTILE = True
    PERCENTILE = 95      # top 10% most salient pixels

    if USE_PERCENTILE:
        thr = np.percentile(OMS_norm, PERCENTILE)
        print(f"Using percentile threshold = {thr:.4f}")
    else:
        thr = 0.4  # fixed threshold
        print(f"Using fixed threshold = {thr}")

    # ------------------------------------------
    # Step 3: Create binary mask of salient pixels
    # ------------------------------------------
    saliency_mask = OMS_norm >= thr

    print(f"Salient area: {np.sum(saliency_mask)} pixels "
        f"(out of {saliency_mask.size}, "
        f"{100*np.mean(saliency_mask):.2f}% of the image)")

    # ------------------------------------------
    # Step 4: Compute bounding box of salient region
    # ------------------------------------------
    ys_sal, xs_sal = np.where(saliency_mask)

    if len(xs_sal) == 0:
        print("WARNING: No salient pixels found. Try lowering threshold.")
        x_min = y_min = 0
        x_max = max_x
        y_max = max_y
    else:
        x_min, x_max = xs_sal.min(), xs_sal.max()
        y_min, y_max = ys_sal.min(), ys_sal.max()

    print("\n🔍 Proposed cropping coordinates:")
    print(f"  x_min={x_min}, x_max={x_max}")
    print(f"  y_min={y_min}, y_max={y_max}")
    print(f"  Crop size = {(y_max-y_min)} x {(x_max-x_min)}")

    # =======================================================
    # Step 4.5: Event Reduction Ratio (ERR) — Crop-based
    # =======================================================

    # Total number of original events
    N_original = len(xs)
    

    # Count events that fall INSIDE the crop
    inside_crop = np.logical_and.reduce((
        xs >= x_min,
        xs <= x_max,
        ys >= y_min,
        ys <= y_max
    ))

    N_filtered = np.sum(inside_crop)
    N_suppressed = N_original - N_filtered

    ERR = 1.0 - (N_filtered / N_original)

    print("\n========== Event Reduction Stats (Crop-based) ==========")
    print(f"Original events   : {N_original}")
    print(f"Events in crop    : {N_filtered}")
    print(f"Suppressed events : {N_suppressed}")
    print(f"ERR               : {ERR:.4f} ({ERR*100:.2f}%)")
    print("=======================================================")


    # ------------------------------------------
    # Step 5: Crop original event image & OMS map
    # ------------------------------------------
    cropped_event_map = window_pos[y_min:y_max+1, x_min:x_max+1]
    cropped_OMS_map = OMS_norm[y_min:y_max+1, x_min:x_max+1]

    # ------------------------------------------
    # Step 6: Visualize cropping results
    # ------------------------------------------
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 3, 1)
    plt.title("Original Event Map")
    plt.imshow(window_pos, cmap="gray")
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

    # You can save the cropping box for later use in model preprocessing:
    crop_box = {
        "x_min": int(x_min),
        "x_max": int(x_max),
        "y_min": int(y_min),
        "y_max": int(y_max)
    }

    print("\n📦 Saved crop box:", crop_box)


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
    graph_img_resized = cv2.resize(draw_graph_with_dots(events_list, suppressed_list, dropped_list),
                                (scaled_width, scaled_height))

    background[:, :scaled_width] = window_pos_resized
    background[:, scaled_width:scaled_width*2] = OMS_resized
    background[:, scaled_width*2:] = graph_img_resized
    

    cv2.putText(background, 'Event map', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,255,0), 2)
    cv2.putText(background, 'OMS map', (scaled_width+30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,255,0), 2)

    cv2.imshow("Visualization", background)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
