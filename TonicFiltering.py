import numpy as np
import argparse
import cv2
import tonic
import torch

# 1. Tonic Filtering based on Inter-Event Interval (IEI)
def filter_tonic_iei(xs_raw, ys_raw, timestamps_raw, pols_raw, max_x, max_y, T_tonic=0.005):
    """
    Suppresses events if the Inter-Event Interval (IEI) is less than T_tonic.
    (This is the hard-threshold tonic filter.)

    Args:
        T_tonic (float): Minimum time difference (in seconds) required to pass the event.
    """
    print(f"Applying IEI Thresholding Filter (T_tonic = {T_tonic * 1000:.1f} ms)")
    
    t_last = np.zeros((max_y, max_x), dtype=np.float64) 
    xs_filtered, ys_filtered, timestamps_filtered, pols_filtered = [], [], [], []

    for x, y, t, p in zip(xs_raw, ys_raw, timestamps_raw, pols_raw):
        if y < max_y and x < max_x:
            t_previous = t_last[y, x]
            delta_t = t - t_previous
            
            # Pass event if IEI is greater than the tonic threshold
            if delta_t > T_tonic:
                xs_filtered.append(x)
                ys_filtered.append(y)
                timestamps_filtered.append(t)
                pols_filtered.append(p)
                t_last[y, x] = t # Update last firing time
                
                
    N_original = len(xs_raw)
    N_filtered = len(xs_filtered)

    ERR = 1.0 - (N_filtered / N_original)

    print("========== Event Reduction Stats ==========")
    print(f"Original events  : {N_original}")
    print(f"Filtered events  : {N_filtered}")
    print(f"Suppressed events: {N_original - N_filtered}")
    print(f"ERR              : {ERR:.4f} ({ERR*100:.2f}%)")
    print("==========================================")
                
    return np.array(xs_filtered), np.array(ys_filtered), np.array(timestamps_filtered), np.array(pols_filtered)

# 2. Dead Time Filter
def filter_tonic_deadtime(xs_raw, ys_raw, timestamps_raw, pols_raw, max_x, max_y, T_refractory=0.005):
    """
    Suppresses all events that arrive within a fixed refractory period 
    after a successful event.

    Args:
        T_refractory (float): Fixed refractory period (in seconds).
    """
    print(f"Applying Dead Time Filter (T_refractory = {T_refractory * 1000:.1f} ms)")

    # Memory stores the time when the pixel is "out of refractory"
    t_ready = np.zeros((max_y, max_x), dtype=np.float64) 
    xs_filtered, ys_filtered, timestamps_filtered, pols_filtered = [], [], [], []

    for x, y, t, p in zip(xs_raw, ys_raw, timestamps_raw, pols_raw):
        if y < max_y and x < max_x:
            t_is_ready = t_ready[y, x]
            
            # Pass event if the current time (t) is AFTER the ready time (t_is_ready)
            if t >= t_is_ready:
                xs_filtered.append(x)
                ys_filtered.append(y)
                timestamps_filtered.append(t)
                pols_filtered.append(p)
                
                # Update t_ready: The pixel is now busy until t + T_refractory
                t_ready[y, x] = t + T_refractory
    N_original = len(xs_raw)
    N_filtered = len(xs_filtered)

    ERR = 1.0 - (N_filtered / N_original)

    print("========== Event Reduction Stats ==========")
    print(f"Original events  : {N_original}")
    print(f"Filtered events  : {N_filtered}")
    print(f"Suppressed events: {N_original - N_filtered}")
    print(f"ERR              : {ERR:.4f} ({ERR*100:.2f}%)")
    print("==========================================")
                
    return np.array(xs_filtered), np.array(ys_filtered), np.array(timestamps_filtered), np.array(pols_filtered)


#3. Basic Leaky Integrate-and-Fire (LIF) Suppressor
def filter_tonic_lif(xs_raw, ys_raw, timestamps_raw, pols_raw, max_x, max_y, tau=0.010, A=1.0, Theta_sat=2.0):
    """
    Filters events based on a basic LIF model, suppressing events if the 
    accumulated membrane potential exceeds a saturation threshold.

    Args:
        tau (float): Time constant for membrane decay (in seconds).
        A (float): Energy added by each event.
        Theta_sat (float): Saturation threshold for suppression.
    """
    print(f"Applying LIF Suppressor Filter (tau = {tau * 1000:.1f} ms, Theta_sat = {Theta_sat})")

    M = np.zeros((max_y, max_x), dtype=np.float32) 
    t_last = np.zeros((max_y, max_x), dtype=np.float64) # Used for decay calculation
    
    xs_filtered, ys_filtered, timestamps_filtered, pols_filtered = [], [], [], []

    for x, y, t, p in zip(xs_raw, ys_raw, timestamps_raw, pols_raw):
        if y < max_y and x < max_x:
            t_previous = t_last[y, x]
            delta_t = t - t_previous
            
            # 1. Decay the potential (Exponential decay)
            # M_decayed = M_previous * exp(-delta_t / tau)
            M_decayed = M[y, x] * np.exp(-delta_t / tau)
            
            # 2. Add new event energy
            M_new = M_decayed + A
            
            # 3. Filtering Decision: Suppress if saturation threshold is hit
            if M_new <= Theta_sat:
                # Pass the event
                xs_filtered.append(x)
                ys_filtered.append(y)
                timestamps_filtered.append(t)
                pols_filtered.append(p)
                
                # Update memory for the passed event
                M[y, x] = M_new
            
            # Even if suppressed, the decay time must be updated
            t_last[y, x] = t
            
    N_original = len(xs_raw)
    N_filtered = len(xs_filtered)

    ERR = 1.0 - (N_filtered / N_original)

    print("========== Event Reduction Stats ==========")
    print(f"Original events  : {N_original}")
    print(f"Filtered events  : {N_filtered}")
    print(f"Suppressed events: {N_original - N_filtered}")
    print(f"ERR              : {ERR:.4f} ({ERR*100:.2f}%)")
    print("==========================================")

    return np.array(xs_filtered), np.array(ys_filtered), np.array(timestamps_filtered), np.array(pols_filtered)

# convert_to_rgb function
def convert_to_rgb(image):
    """
    Converts a single-channel image (like an event map) to a 3-channel RGB image.
    If the image is already 3-channel, it returns it unchanged.
    """
    # Check if the image has only 2 dimensions (Grayscale/single channel)
    if len(image.shape) == 2:
        # Convert the grayscale image to BGR (OpenCV's default color space)
        # BGR is used here because cv2.imshow expects BGR
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) 
    else:
        return image
    
# --- Example Parser Setup ---
parser = argparse.ArgumentParser(description='Event filtering pipeline with tonic options.')
parser.add_argument(
    '--tonic_filter', 
    type=str, 
    default='IEI', 
    choices=['NONE', 'IEI', 'DEADTIME', 'LIF'],
    help='Select the tonic filtering technique to apply.'
)
args = parser.parse_args()
# --------------------

if __name__ == "__main__":
    
    
    # ---------------------------
    # 1. Dsec dataset loading
    # ---------------------------
    data = np.loadtxt('Dsec.txt')

    xs_raw = data[:, 0].astype(int)
    ys_raw = data[:, 1].astype(int)
    timestamps_raw = data[:, 2]
    pols_raw = data[:, 3].astype(int)
    # you have to put the scale_factor=0.75

    """
    # ---------------------------
    # Load DVS Gesture dataset
    # ---------------------------
    
    data = tonic.datasets.DVSGesture(save_to="../Datasets", train=True)
    events, targets = data[0]  # Get the first sample
    xs_raw = events["x"].astype(int)
    ys_raw = events["y"].astype(int)    
    pols_raw = events["p"].astype(int)
    timestamps_raw = events["t"]

    # you have to put the scale_factor=3
    """

    max_x = int(np.max(xs_raw)) + 1
    max_y = int(np.max(ys_raw)) + 1

    # Accumulate RAW Events for Baseline Visualization
    window_pos_raw = np.zeros((max_y, max_x), dtype=np.uint16)
    for x, y, p in zip(xs_raw, ys_raw, pols_raw):
        if y < max_y and x < max_x and p == 1:
            window_pos_raw[y, x] += 1
    # -----------------------------------
    
    # -----------------------------------
    # 2. Apply Selected Tonic Filter
    # -----------------------------------
    xs, ys, timestamps, pols = xs_raw, ys_raw, timestamps_raw, pols_raw # Default to raw if NONE selected

    if args.tonic_filter == 'IEI':
        xs, ys, timestamps, pols = filter_tonic_iei(xs_raw, ys_raw, timestamps_raw, pols_raw, max_x, max_y, T_tonic=0.005)
    elif args.tonic_filter == 'DEADTIME':
        xs, ys, timestamps, pols = filter_tonic_deadtime(xs_raw, ys_raw, timestamps_raw, pols_raw, max_x, max_y, T_refractory=0.008)
    elif args.tonic_filter == 'LIF':
        xs, ys, timestamps, pols = filter_tonic_lif(xs_raw, ys_raw, timestamps_raw, pols_raw, max_x, max_y, tau=0.015, Theta_sat=3.0)
    elif args.tonic_filter == 'NONE':
        print("Skipping Tonic Filtering.")
    
    print(f"Original events: {len(xs_raw)}. Filtered events: {len(xs)}")
    
    # -----------------------------------
    # 3. Accumulate TONIC Filtered Events
    # -----------------------------------
    window_pos_tonic = np.zeros((max_y, max_x), dtype=np.uint16) 
    
    for x, y, p in zip(xs, ys, pols):
        if y < max_y and x < max_x and p == 1:
            window_pos_tonic[y, x] += 1
    # -----------------------------------
    
    # -----------------------------------
    # 4. Visualization
    # -----------------------------------
    scale_factor = 0.7
    scaled_height = int(max_y * scale_factor)
    scaled_width = int(max_x * scale_factor)

    # Visualization will now show 2 maps side-by-side
    background = np.ones((scaled_height, scaled_width * 2, 3), dtype=np.uint8) * 255

    # 1. RAW Event Map
    window_pos_raw_resized = convert_to_rgb(cv2.resize(window_pos_raw, (scaled_width, scaled_height)))
    
    # 2. TONIC Filtered Event Map
    window_pos_tonic_resized = convert_to_rgb(cv2.resize(window_pos_tonic, (scaled_width, scaled_height)))

    # Place images
    background[:, :scaled_width] = window_pos_raw_resized
    background[:, scaled_width:] = window_pos_tonic_resized

    # Text Labels
    cv2.putText(background, 'RAW Event Map (Baseline)', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    cv2.putText(background, f'TONIC Filtered Map ({args.tonic_filter})', (scaled_width + 30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

    cv2.imshow("Tonic Filtering Visualization", background)
    cv2.waitKey(0)
    cv2.destroyAllWindows()