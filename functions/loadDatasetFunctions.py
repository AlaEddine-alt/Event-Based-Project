import numpy as np
import tonic

def load_events(dataset_name):

    if dataset_name == "DSEC":
        data = np.loadtxt('Dsec.txt')  # columns: x y t p
        xs = data[:, 0].astype(int)
        ys = data[:, 1].astype(int)
        timestamps = data[:, 2]         
        pols = data[:, 3].astype(int)
        scale_factor = 0.75  # for DSEC   
        
    elif dataset_name == "DVSGesture":
        dataset = tonic.datasets.DVSGesture(save_to="../Datasets", train=True)
        events, target = dataset[0]
        xs = events["x"].astype(int)
        ys = events["y"].astype(int)
        pols = events["p"].astype(int)
        timestamps = events["t"]
        scale_factor = 3 # for DVSGesture
    else:
        raise ValueError("Unsupported dataset")
    
    
    
    return xs, ys, timestamps, pols, scale_factor

def reset_windows(xs, ys, pols):
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

    return window_pos, window_neg, max_x, max_y, numevs

    