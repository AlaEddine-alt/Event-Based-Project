import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights
import torch.nn.functional as F
import os
import time
import numpy as np
from torch.utils.data import Dataset

from functions.loadDatasetFunctions import DVSGestureNPYDataset
from functions.saveAndLoadFilteredData import FilteredNPYDataset
from Classification.ComplexCNN import  VoxelGridConverter, ModelTrainer
from torch.utils.data import DataLoader


class PretrainedRes3D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # R3D_18 handles (Batch, Channel, Time, Height, Width)
        self.backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)
        num_fts = self.backbone.fc.in_features
        self.conv0 = nn.Conv2d(2, 3, kernel_size=1)  # Convert 2 channels to 3 for compatibility with Conv3D

        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_fts, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)  # Merge batch and time for conv0
        x = self.conv0(x)  # (B*T, 3, H, W)
        x = x.view(B, T, 3, H, W)
        # Input (B, T, C, H, W) -> Output (B, C, T, H, W)
        return self.backbone(x.transpose(1, 2))


class Custom3DCNN(nn.Module):
    def __init__(self, num_classes, num_frames=5):
        super(Custom3DCNN, self).__init__()
        
        self.conv0 = nn.Conv2d(2, 3, kernel_size=1)  # Convert 2 channels to 3 for compatibility with Conv3D

        # Layer 1: Temporal & Spatial feature extraction
        self.conv1 = nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(16)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2)) # Reduce spatial, keep temporal
        
        # Layer 2: Deeper features
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(32)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2)) # Reduce both
        
        # Layer 3: High-level features
        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(64)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        # Layer 4: High-level features
        #self.conv4 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        #self.bn4 = nn.BatchNorm3d(128)
        #self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        # Global Average Pooling to make it input-size agnostic
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Input: (B, T, C, H, W) -> Change to (B, C, T, H, W) for Conv3D
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)  # Merge batch and time for conv0
        x = self.conv0(x)  # (B*T, 3, H, W)
        x = x.view(B, T, 3, H, W)  # Reshape back to (B, T, C, H, W)
        #print(f"Input shape: {x.shape}")
        x = x.transpose(1, 2)
        
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        #x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    



# --- RESNET 3D COMPONENTS ---

class ResidualBlock3D(nn.Module):
    """
    Standard 3D Residual Block with an identity shortcut.
    If stride > 1 or channel count changes, a 1x1x1 projection is used in the shortcut.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock3D, self).__init__()
        # First 3D Conv
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                               stride=(1, stride, stride), padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        # Second 3D Conv
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                          stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet3D_Custom(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super(ResNet3D_Custom, self).__init__()
        
        # Initial Stage: Conv1 (7x7) + MaxPool
        # Reduces spatial dim from 128 -> 32 while keeping temporal resolution
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=(3, 7, 7), 
                               stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        # Layer 1: 2x Blocks, 64 channels, output (32x32)
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        
        # Layer 2: 2x Blocks, 128 channels, output (16x16)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        
        # Layer 3: 2x Blocks, 256 channels, output (8x8)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)

        # Global Avg Pool and Classifier
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_planes, out_planes, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock3D(in_planes, out_planes, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock3D(out_planes, out_planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Input shape: (Batch, Time, Channels, H, W) -> (B, 3, T, H, W)
        x = x.transpose(1, 2)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)
    
def generate_voxel_grid(events, shape, nr_temporal_bins, separate_pol=True, normalize_event=False):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.
    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param nr_temporal_bins: number of bins in the temporal axis of the voxel grid
    :param shape: dimensions of the voxel grid
    """
    height, width = shape
    
    #assert(events.shape[1] == 4)
    assert(nr_temporal_bins > 0)
    assert(width > 0)
    assert(height > 0)

    #events = events[np.argsort(events['t'])]


    voxel_grid_positive = np.zeros((nr_temporal_bins, height, width), np.float32).ravel()
    voxel_grid_negative = np.zeros((nr_temporal_bins, height, width), np.float32).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events['t'][-1] #events[-1, 2] 
    first_stamp = events['t'][0] #last_stamp - 300000000 #events[0, 2]
    deltaT = last_stamp - first_stamp
    #print(f"actual: First stamp: {events[0, 2]}\t Last stamp: {events[-1, 2]}\n")
    #print(f"choosen: First stamp: {first_stamp}\t Last stamp: {last_stamp}\n")

    if deltaT == 0:
        deltaT = 1.0

    # events[:, 2] = (nr_temporal_bins - 1) * (events[:, 2] - first_stamp) / deltaT
    #xs = events[:, 0].astype(np.int64)
    #ys = events[:, 1].astype(np.int64)

    xs = events['x'].astype(np.int64)
    ys = events['y'].astype(np.int64)
    # ts = events[:, 2]
    # print(ts[:10])
    ts = (nr_temporal_bins - 1) * (events['t'] - first_stamp) / deltaT

    pols = events['p']
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(np.int64)
    dts = ts - tis
    vals_left = np.abs(pols) * (1.0 - dts)
    vals_right = np.abs(pols) * dts
    pos_events_indices = pols == 1

    # Positive Voxels Grid
    valid_indices_pos = np.logical_and(tis < nr_temporal_bins, pos_events_indices)
    valid_pos = (xs < width) & (xs >= 0) & (ys < height) & (ys >= 0) & (ts >= 0) & (ts < nr_temporal_bins)
    valid_indices_pos = np.logical_and(valid_indices_pos, valid_pos)

    np.add.at(voxel_grid_positive, xs[valid_indices_pos] + ys[valid_indices_pos] * width +
              tis[valid_indices_pos] * width * height, vals_left[valid_indices_pos])

    valid_indices_pos = np.logical_and((tis + 1) < nr_temporal_bins, pos_events_indices)
    valid_indices_pos = np.logical_and(valid_indices_pos, valid_pos)
    np.add.at(voxel_grid_positive, xs[valid_indices_pos] + ys[valid_indices_pos] * width +
              (tis[valid_indices_pos] + 1) * width * height, vals_right[valid_indices_pos])

    # Negative Voxels Grid
    valid_indices_neg = np.logical_and(tis < nr_temporal_bins, ~pos_events_indices)
    valid_indices_neg = np.logical_and(valid_indices_neg, valid_pos)

    np.add.at(voxel_grid_negative, xs[valid_indices_neg] + ys[valid_indices_neg] * width +
              tis[valid_indices_neg] * width * height, vals_left[valid_indices_neg])

    valid_indices_neg = np.logical_and((tis + 1) < nr_temporal_bins, ~pos_events_indices)
    valid_indices_neg = np.logical_and(valid_indices_neg, valid_pos)
    np.add.at(voxel_grid_negative, xs[valid_indices_neg] + ys[valid_indices_neg] * width +
              (tis[valid_indices_neg] + 1) * width * height, vals_right[valid_indices_neg])

    voxel_grid_positive = np.reshape(voxel_grid_positive, (nr_temporal_bins, height, width))
    voxel_grid_negative = np.reshape(voxel_grid_negative, (nr_temporal_bins, height, width))

    if separate_pol:
        if normalize_event:
            fn = normalize_voxel_grid_numpy
        else:
            fn = lambda x:x
        return fn(np.concatenate([voxel_grid_positive, voxel_grid_negative], axis=0))

    voxel_grid = voxel_grid_positive - voxel_grid_negative
    if normalize_event:
        voxel_grid = normalize_voxel_grid_numpy(voxel_grid)
    return voxel_grid

def normalize_voxel_grid_numpy(voxel_grid):
    """Normalize event voxel grids"""
    mask = np.nonzero(voxel_grid)
    if mask[0].shape[0] > 0:
        mean = voxel_grid[mask].mean()
        std = voxel_grid[mask].std()
        if std > 0:
            voxel_grid[mask] = (voxel_grid[mask] - mean) / std
        else:
            voxel_grid[mask] = voxel_grid[mask] - mean
    return voxel_grid


class DVSGestureDataset(Dataset):
    def __init__(self, event_data, labels, event_polarity=True, precompute=True):
        self.labels = labels
        self.event_polarity = event_polarity
        self.data = []
        if precompute:
            print(f"Pre-computing {len(event_data)} representations...")
            #self.data = [generate_voxel_grid(events, shape=(128, 128), nr_temporal_bins=5) for events in event_data]
            for events in event_data:
                temp = generate_voxel_grid(events, shape=(128, 128), nr_temporal_bins=5)
                temp = torch.from_numpy(temp)

                if self.event_polarity:
                    # Split slices into two groups pos and neg
                    pos, neg = torch.chunk(temp, 2, dim=0) 
                    
                    temp = torch.stack([pos, neg], dim=1)
                self.data.append(temp)
            self.event_data = None
        else:
            self.data = None
            self.event_data = event_data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        if self.data is not None:
            representation = self.data[idx]
        else:
            representation = generate_voxel_grid(self.event_data[idx], shape=(128, 128), nr_temporal_bins=5)
            representation = torch.from_numpy(representation)

            if self.event_polarity:
                # Split slices into two groups pos and neg
                pos, neg = torch.chunk(representation, 2, dim=0) 
                
                representation = torch.stack([pos, neg], dim=1)
        return representation, label

    
def train_model(dataset_training, dataset_testing):
    
    # Extract events and labels
    def extract_events(dataset):
        events_list, labels_list = [], []
        for i in range(len(dataset)):
            events, label = dataset[i]
            events_list.append({'x': events['x'], 'y': events['y'], 't': events['t'], 'p': events['p']})
            labels_list.append(label)
        return events_list, labels_list

    train_events, train_labels = extract_events(dataset_training)
    test_events, test_labels = extract_events(dataset_testing)

    # Choose converter
    
    #converter = EventFrameConverter(height=128, width=128)
    #converter = StackedFrameConverter(128, 128, num_frames=5)
    #converter = TimeSurfaceConverter(128, 128, tau=50000)
    # converter = VoxelGridConverter(128, 128, num_bins=5)
    
    #dummy_event = {'x': np.array([0]), 'y': np.array([0]), 't': np.array([0]), 'p': np.array([1])}
    #num_channels = converter.convert(dummy_event).shape[0]
    #print(f"Using converter {converter.__class__.__name__} with {num_channels} channels")

    # Prepare datasets & loaders
    train_dataset = DVSGestureDataset(train_events, train_labels, precompute=True)
    test_dataset = DVSGestureDataset(test_events, test_labels, precompute=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Build model
    model = Custom3DCNN(num_classes=11)
    trainer = ModelTrainer(model)
    start_time = time.time()
    best_accuracy = trainer.train(train_loader, test_loader, num_epochs=20, lr=0.001)
    end_time = time.time()
    time_elapsed = end_time - start_time
    print(f"Training + evaluation time: {time_elapsed:.2f} seconds")

    # Confusion matrix
    class_names = ["Hand Clapping", "Right Hand Wave", "Left Hand Wave",
                   "Right Arm CW", "Right Arm CCW", "Left Arm CW", "Left Arm CCW",
                   "Arm Roll", "Air Drums", "Air Guitar", "Other"]
    # trainer.plot_confusion_matrix(test_loader, class_names)


    return best_accuracy, time_elapsed


if __name__ == "__main__":

    training_ROOT = "C:/Users/giuli/Desktop/Giulia/PER/Event-Based-Project/Datasets/FilteredDatasets/OMS/train"
    testing_ROOT = "C:/Users/giuli/Desktop/Giulia/PER/Event-Based-Project/Datasets/FilteredDatasets/OMS/test"

    # Downsapled dataset 
    #training_ROOT = "C:/Users/giuli/Desktop/Giulia/PER/Event-Based-Project/DVSGestureDownsampled/ibmGestureTrain"
    #testing_ROOT = "C:/Users/giuli/Desktop/Giulia/PER/Event-Based-Project/DVSGestureDownsampled/ibmGestureTest"

    training_users = sorted(os.listdir(training_ROOT))
    test_users = sorted(os.listdir(testing_ROOT))


    # Training with raw downsampled dataset 

    #train_dataset_raw = DVSGestureNPYDataset(training_ROOT, users=training_users)
    #test_dataset_raw = DVSGestureNPYDataset(testing_ROOT, users=test_users)

    train_dataset_raw = FilteredNPYDataset("Datasets/FilteredDatasets/Attention/train")
    test_dataset_raw = FilteredNPYDataset("Datasets/FilteredDatasets/Attention/test")
    
    acc_raw, time_training_raw = train_model(train_dataset_raw, test_dataset_raw)
    print(f"\nTime taken for training the ComplexCNN model: {time_training_raw:.2f} seconds")
    print(f"Best Test Accuracy: {acc_raw:.2f}%")

    """
    Best Test Accuracy: 42.58%
    Did not reach 95.0% accuracy
    Training + evaluation time: 383.84 seconds

    Time taken for training the ComplexCNN model: 383.84 seconds
    Best Test Accuracy: 42.58%
    """

