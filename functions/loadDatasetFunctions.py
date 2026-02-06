import numpy as np
import tonic

import os
import numpy as np
from torch.utils.data import Dataset

class DVSGestureNPYDataset(Dataset):
    def __init__(self, root_dir, users=None):
        self.samples = []

        for user_folder in sorted(os.listdir(root_dir)):
            if users is not None and user_folder not in users:
                continue

            folder_path = os.path.join(root_dir, user_folder)
            if not os.path.isdir(folder_path):
                continue

            for label in range(11):
                npy_path = os.path.join(folder_path, f"{label}.npy")
                if not os.path.exists(npy_path):
                    continue
                self.samples.append((npy_path, label))

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def _load_events(self, path):
        data = np.load(path)

        # structured array
        if data.dtype.names is not None:
            return {
                'x': data['x'],
                'y': data['y'],
                't': data['t'],
                'p': data['p']
            }

        # plain Nx4 array
        return {
            'x': data[:, 0],
            'y': data[:, 1],
            't': data[:, 2],
            'p': data[:, 3]
        }

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        events = self._load_events(path)
        return events, label


def load_events(dataset_name):

    if dataset_name == "DSEC":
        data = np.loadtxt('Dsec.txt')  # columns: x y t p
        xs = data[:, 0].astype(int)
        ys = data[:, 1].astype(int)
        timestamps = data[:, 2]         
        pols = data[:, 3].astype(int)
        scale_factor = 0.75  # for DSEC   
        return xs, ys, timestamps, pols, scale_factor
        
    elif dataset_name == "DVSGesture":
        dataset = tonic.datasets.DVSGesture(save_to="../Datasets", train=True)
        event_list, target_list = extract_events(dataset)
        scale_factor = 3 # for DVSGesture
        return event_list, target_list, scale_factor
    else:
        raise ValueError("Unsupported dataset")
    

# Extract events and labels
def extract_events(dataset):
    events_list, labels_list = [], []
    for i in range(len(dataset)):
        events, label = dataset[i]
        events_list.append({'x': events['x'], 'y': events['y'], 't': events['t'], 'p': events['p']})
        labels_list.append(label)
    return events_list, labels_list

def extract_single_event(event):
    xs = event["x"].astype(int)
    ys = event["y"].astype(int)
    pols = event["p"].astype(int)
    timestamps = event["t"]
    return xs, ys, timestamps, pols

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

    