import os
import numpy as np
from torch.utils.data import Dataset

class FilteredNPZDataset(Dataset):
    def __init__(self, root_dir):
        self.files = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".npz")
        ])

        if len(self.files) == 0:
            raise RuntimeError(f"No .npz files found in {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])

        events = {
            'x': data['x'],
            'y': data['y'],
            't': data['t'],
            'p': data['p']
        }

        label = int(data['label'])
        return events, label


def save_filtered_dataset(dataset, save_dir, prefix="sample"):
    """
    dataset: list or Dataset of (events_dict, label)
    save_dir: target directory
    prefix: file prefix (e.g. train / test)
    """
    os.makedirs(save_dir, exist_ok=True)

    for idx, (events, label) in enumerate(dataset):
        save_path = os.path.join(save_dir, f"{prefix}_{idx:05d}.npz")

        np.savez_compressed(
            save_path,
            x=events['x'],
            y=events['y'],
            t=events['t'],
            p=events['p'],
            label=label
        )

    print(f"Saved {len(dataset)} samples to {save_dir}")
