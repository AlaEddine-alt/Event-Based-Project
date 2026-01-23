import cv2
import numpy as np
import torch
import tonic
import matplotlib.pyplot as plt

from functions.OMS_helpers import *
from functions.attention_helpers import AttentionModule

# ---------------------------
# Config
# ---------------------------
class Config:
    RESOLUTION = [128, 128]
    MAX_X = RESOLUTION[0]
    MAX_Y = RESOLUTION[1]

    DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    GLOBAL_PERCENTILE = 95
    INSTANCE_PADDING = 10

    FIXED_OUTPUT_SIZE = True
    OUTPUT_SIZE = (64, 64)

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
        'VM_radius': 8,
        'VM_radius_group': 15,
        'num_ori': 4,
        'b_inh': 3,
        'g_inh': 1.0,
        'w_sum': 0.5,
        'vm_w': 0.2,
        'vm_w2': 0.4,
        'vm_w_group': 0.2,
        'vm_w2_group': 0.4,
        'random_init': False,
        'lif_tau': 0.3
    }

config = Config()

# ---------------------------
# Helper functions
# ---------------------------
def compute_OMS(window_pos):
    OMSpos = torch.tensor(window_pos, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)

    OMS_map, _ = egomotion(
        OMSpos,
        net_center,
        net_surround,
        config.DEVICE,
        config.MAX_Y,
        config.MAX_X,
        config.OMS_PARAMS['threshold']
    )

    return OMS_map.squeeze().detach().cpu().numpy()


def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-6)


def get_bbox_from_mask(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return xs.min(), xs.max(), ys.min(), ys.max()


def clamp(val, minv, maxv):
    return max(minv, min(val, maxv))


# ---------------------------
# Load DVSGesture sample
# ---------------------------
dataset = tonic.datasets.DVSGesture(save_to="../Datasets", train=True)
events, label = dataset[0]

# ORIGINAL EVENT COUNT
N_original = len(events)

xs = events["x"].astype(int)
ys = events["y"].astype(int)
pols = events["p"].astype(int)

max_x = int(xs.max()) + 1
max_y = int(ys.max()) + 1

window_pos = np.zeros((max_y, max_x), dtype=np.uint16)

for x, y, p in zip(xs, ys, pols):
    if p == 1:
        window_pos[y, x] += 1

# ---------------------------
# Initialize models
# ---------------------------
net_center, net_surround = initialize_oms(config.DEVICE, config.OMS_PARAMS)
net_attention = AttentionModule(**config.ATTENTION_PARAMS)

# ======================================================
# SCENARIO 3 — GLOBAL + INSTANCE ROI
# ======================================================
OMS_map = compute_OMS(window_pos)
OMS_norm = normalize(OMS_map)

thr = np.percentile(OMS_norm, config.GLOBAL_PERCENTILE)
global_mask = OMS_norm >= thr
global_bbox = get_bbox_from_mask(global_mask)

if global_bbox is None:
    raise RuntimeError("No global saliency region found!")

gx_min, gx_max, gy_min, gy_max = global_bbox
global_crop = window_pos[gy_min:gy_max+1, gx_min:gx_max+1]

activity_mask = global_crop > 0
instance_bbox = get_bbox_from_mask(activity_mask)

if instance_bbox is None:
    raise RuntimeError("No activity inside global ROI!")

ix_min, ix_max, iy_min, iy_max = instance_bbox

pad = config.INSTANCE_PADDING
ix_min = clamp(ix_min - pad, 0, global_crop.shape[1] - 1)
ix_max = clamp(ix_max + pad, 0, global_crop.shape[1] - 1)
iy_min = clamp(iy_min - pad, 0, global_crop.shape[0] - 1)
iy_max = clamp(iy_max + pad, 0, global_crop.shape[0] - 1)

final_crop = global_crop[iy_min:iy_max+1, ix_min:ix_max+1]

if config.FIXED_OUTPUT_SIZE:
    final_crop = cv2.resize(
        final_crop,
        config.OUTPUT_SIZE,
        interpolation=cv2.INTER_NEAREST
    )

# ---------------------------
# EVENT REDUCTION RATIO (ERR)
# ---------------------------
final_x_min = gx_min + ix_min
final_x_max = gx_min + ix_max
final_y_min = gy_min + iy_min
final_y_max = gy_min + iy_max

filtered_mask = (
    (events["x"] >= final_x_min) &
    (events["x"] <= final_x_max) &
    (events["y"] >= final_y_min) &
    (events["y"] <= final_y_max)
)

N_filtered = np.sum(filtered_mask)
ERR = 1.0 - (N_filtered / N_original)

# ---------------------------
# VISUALIZATION
# ---------------------------
plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.title("Original Event Map")
plt.imshow(window_pos, cmap="gray")

plt.subplot(1, 4, 2)
plt.title("OMS Saliency")
plt.imshow(OMS_norm, cmap="jet")

plt.subplot(1, 4, 3)
plt.title("Global ROI")
plt.imshow(global_crop, cmap="gray")

plt.subplot(1, 4, 4)
plt.title("Final Combined Crop")
plt.imshow(final_crop, cmap="gray")

plt.tight_layout()
plt.show()

# ---------------------------
# OUTPUT
# ---------------------------
print("\n📊 Event Reduction Ratio (Scenario 3)")
print(f"Original events: {N_original}")
print(f"Filtered events: {N_filtered}")
print(f"ERR = {ERR:.4f}")
print("Label:", label)