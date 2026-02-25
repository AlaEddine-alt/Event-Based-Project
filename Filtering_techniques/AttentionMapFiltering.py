import cv2
import numpy as np
import torch

from functions.attention_helpers import AttentionModule
from functions.visualizationFunctions import draw_graph_with_dots, convert_to_rgb
from functions.loadDatasetFunctions import extract_single_event, reset_windows
from functions.adaptFilteredData import tuple_events_to_event_dict


# ---------------------------
# Config
# ---------------------------
class Config:
    RESOLUTION = [128, 128]
    MAX_X = RESOLUTION[0]
    MAX_Y = RESOLUTION[1]
    DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

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


# ---------------------------
# Attention Filtering
# ---------------------------
class AttentionFiltering:

    def __init__(self, event, scale_factor):

        xs, ys, timestamps, pols = extract_single_event(event)
        window_pos, window_neg, max_x, max_y, numevs = reset_windows(xs, ys, pols)

        self.xs = xs
        self.ys = ys
        self.timestamps = timestamps
        self.pols = pols

        self.window_pos = window_pos
        self.window_neg = window_neg

        self.max_x = max_x
        self.max_y = max_y
        self.numevs = numevs

        self.scale_factor = scale_factor

        self.events_list = [numevs[0]]
        self.suppressed_list = [0]
        self.dropped_list = [0]

        self.config = Config()

    # --------------------------------------------------
    # Main Attention-Based Filtering
    # --------------------------------------------------
    def Attention_filtering(self):

        # Initialize Attention Module
        net_attention = AttentionModule(**self.config.ATTENTION_PARAMS).to(self.config.DEVICE)
        net_attention.eval()

        # Prepare Input Tensor [1, 2, H, W]
        att_in = np.stack([self.window_pos, self.window_neg], axis=0)
        att_in = torch.from_numpy(att_in).float().to(self.config.DEVICE)
        att_in = att_in.unsqueeze(0)

        # Forward Pass
        with torch.no_grad():
            att_out = net_attention(att_in)

        S = att_out.squeeze().cpu().numpy()

        # Normalize Saliency Map
        min_val = np.min(S)
        max_val = np.max(S)

        if max_val > min_val:
            S_normalized = (S - min_val) / (max_val - min_val)
        else:
            S_normalized = np.ones_like(S)

        # Threshold 
        saliency_threshold = np.mean(S_normalized)

        # Event-Level Filtering
        filtered_events = []
        suppressed_count = 0

        for x, y, t, p in zip(self.xs, self.ys, self.timestamps, self.pols):

            if y < S_normalized.shape[0] and x < S_normalized.shape[1]:

                if S_normalized[y, x] >= saliency_threshold:
                    filtered_events.append((x, y, t, p))
                else:
                    suppressed_count += 1


        num_total_events = len(self.xs)
        num_filtered_events = len(filtered_events)

        ERR = 1.0 - (num_filtered_events / num_total_events)

        self.suppressed_list.append(suppressed_count)

        events_dict = tuple_events_to_event_dict(filtered_events)

        return events_dict, S_normalized, ERR

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------
    def Attention_visualization(self, saliency_map):

        print("Event dtype:", self.window_pos.dtype)
        print("Event min:", self.window_pos.min())
        print("Event max:", self.window_pos.max())
        print("Event non-zero:", np.count_nonzero(self.window_pos))

        scaled_height = int(self.max_y * self.scale_factor)
        scaled_width = int(self.max_x * self.scale_factor)

        background = np.ones((scaled_height, scaled_width * 2, 3), dtype=np.uint8) * 255

        # Event map (come OMS)
        event_img = convert_to_rgb(
            cv2.resize(self.window_pos,
                    (scaled_width, scaled_height),
                    interpolation=cv2.INTER_NEAREST)
        )

        # Normalize saliency safely
        saliency_8bit = cv2.normalize(
            saliency_map,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        saliency_img = convert_to_rgb(
            cv2.resize(saliency_8bit,
                    (scaled_width, scaled_height),
                    interpolation=cv2.INTER_NEAREST)
        )

        background[:, :scaled_width] = event_img
        background[:, scaled_width:] = saliency_img

        cv2.putText(background, 'Event Map', (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 2)

        cv2.putText(background, 'Attention', (scaled_width + 30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 2)

        cv2.imshow("Attention Visualization", background)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
