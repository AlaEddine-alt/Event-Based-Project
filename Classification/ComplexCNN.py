import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import tonic
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import time

# ------------------------
# Event Converters
# ------------------------
class EventFrameConverter:
    """Single frame (fast, no temporal info)"""
    def __init__(self, height=128, width=128):
        self.height = height
        self.width = width
    
    def convert(self, events):
        if len(events['x']) == 0:
            return torch.zeros((2, self.height, self.width), dtype=torch.float32)
        x = events['x'].astype(np.int32)
        y = events['y'].astype(np.int32)
        p = events['p'].astype(np.int32)
        valid = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        x, y, p = x[valid], y[valid], p[valid]
        frame = np.zeros((2, self.height, self.width), dtype=np.float32)
        np.add.at(frame[0], (y[p == 1], x[p == 1]), 1)
        np.add.at(frame[1], (y[p == 0], x[p == 0]), 1)
        return torch.from_numpy(frame)

class StackedFrameConverter:
    """Split events into N temporal frames"""
    def __init__(self, height=128, width=128, num_frames=5):
        self.height = height
        self.width = width
        self.num_frames = num_frames
    
    def convert(self, events):
        if len(events['x']) == 0:
            return torch.zeros((self.num_frames*2, self.height, self.width), dtype=torch.float32)
        x = events['x'].astype(np.int32)
        y = events['y'].astype(np.int32)
        t = events['t']
        p = events['p'].astype(np.int32)
        valid = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        x, y, t, p = x[valid], y[valid], t[valid], p[valid]
        t_norm = (t - t.min()) / (t.max() - t.min() + 1e-6)
        t_idx = (t_norm * (self.num_frames - 1)).astype(np.int32)
        frames = np.zeros((self.num_frames*2, self.height, self.width), dtype=np.float32)
        for i in range(self.num_frames):
            mask = t_idx == i
            if mask.any():
                x_i, y_i, p_i = x[mask], y[mask], p[mask]
                np.add.at(frames[i*2], (y_i[p_i==1], x_i[p_i==1]), 1)
                np.add.at(frames[i*2+1], (y_i[p_i==0], x_i[p_i==0]), 1)
        return torch.from_numpy(frames)
    

class TimeSurfaceConverter:
    # Time surface: last event timestamp per pixel
    def __init__(self, height=128, width=128, tau=50000):
        self.height = height
        self.width = width
        self.tau = tau
    
    def convert(self, events):
        if len(events['x']) == 0:
            return torch.zeros((2, self.height, self.width), dtype=torch.float32)

        x = events['x'].astype(np.int32)
        y = events['y'].astype(np.int32)
        t = events['t']
        p = events['p']

        valid = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        x, y, t, p = x[valid], y[valid], t[valid], p[valid]

        # force polarity into {0,1}
        p = (p > 0).astype(np.int32)

        surface = np.zeros((2, self.height, self.width), dtype=np.float32)
        t_max = t.max()

        for i in range(len(x)):
            dt = t_max - t[i]
            surface[p[i], y[i], x[i]] = np.exp(-dt / self.tau)

        return torch.from_numpy(surface)


class VoxelGridConverter:
    """Voxel grid: temporal bins, polarity weighted"""
    def __init__(self, height=128, width=128, num_bins=5):
        self.height = height
        self.width = width
        self.num_bins = num_bins
    
    def convert(self, events):
        if len(events['x']) == 0:
            return torch.zeros((self.num_bins, self.height, self.width), dtype=torch.float32)
        x = events['x'].astype(np.int32)
        y = events['y'].astype(np.int32)
        t = events['t']
        p = events['p'].astype(np.int32)
        valid = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        x, y, t, p = x[valid], y[valid], t[valid], p[valid]
        t_norm = (t - t.min()) / (t.max() - t.min() + 1e-6)
        t_idx = (t_norm * (self.num_bins-1)).astype(np.int32)
        voxel = np.zeros((self.num_bins, self.height, self.width), dtype=np.float32)
        polarity_values = 2*p - 1
        np.add.at(voxel, (t_idx, y, x), polarity_values)
        return torch.from_numpy(voxel)

# ------------------------
# Dataset Wrapper
# ------------------------
class DVSGestureDataset(Dataset):
    def __init__(self, event_data, labels, converter, precompute=True):
        self.labels = labels
        self.converter = converter
        if precompute:
            print(f"Pre-computing {len(event_data)} representations...")
            self.data = [self.converter.convert(events) for events in event_data]
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
            representation = self.converter.convert(self.event_data[idx])
        return representation, label

# ------------------------
# CNN Model
# ------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class DVSGestureCNN(nn.Module):
    def __init__(self, num_input_channels, num_classes=11, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_c, out_c, num_blocks, stride):
        layers = [ResidualBlock(in_c, out_c, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_c, out_c))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ------------------------
# Trainer
# ------------------------
class ModelTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', stability_window=5):
        torch.cuda.empty_cache()
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.stability_window = stability_window

    def train_epoch(self, train_loader, optimizer):
        self.model.train()
        total_loss, batch_losses, correct, total = 0, [], 0, 0
        for data, labels in train_loader:
            data, labels = data.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_losses.append(loss.item())
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        return total_loss/len(train_loader), 100.*correct/total, np.std(batch_losses)

    def evaluate(self, test_loader):
        self.model.eval()
        total_loss, batch_losses, correct, total = 0, [], 0, 0
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                batch_losses.append(loss.item())
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        return total_loss/len(test_loader), 100.*correct/total, np.std(batch_losses)

    def train(self, train_loader, validation_loader, num_epochs=50, lr=0.001, convergence_threshold=95.0):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
        best_acc, convergence_epoch = 0, None
        history = {'train_loss': [], 'train_acc': [], 'train_loss_std': [],
                   'test_loss': [], 'test_acc': [], 'test_loss_std': []}

        for epoch in range(num_epochs):
            train_loss, train_acc, train_loss_std = self.train_epoch(train_loader, optimizer)
            val_loss, val_acc, val_loss_std = self.evaluate(validation_loader)
            scheduler.step()
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_loss_std'].append(train_loss_std)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_loss_std'].append(val_loss_std)

            print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.2f}% | Test Acc: {val_acc:.2f}%")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')
            if convergence_epoch is None and val_acc >= convergence_threshold:
                convergence_epoch = epoch + 1

        print(f"Best Validation Accuracy: {best_acc:.2f}%")
        if convergence_epoch: print(f"Converged at epoch {convergence_epoch}")
        else: print(f"Did not reach {convergence_threshold}% accuracy")
        return best_acc

    # Confusion Matrix
    def plot_confusion_matrix(self, dataloader, class_names):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for data, labels in dataloader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots(figsize=(8, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
        plt.show()


def compute_sparsity(dataloader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    total_elements = 0
    total_nonzero = 0
    
    for data, _ in dataloader:
        data = data.to(device)
        
        total_elements += data.numel()
        total_nonzero += torch.count_nonzero(data).item()
    
    sparsity = 1 - (total_nonzero / total_elements)
    return sparsity

# ------------------------
# Main Execution
# ------------------------
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
    

    train_events, validation_events, train_labels, validation_labels = train_test_split(
        train_events, train_labels,
        test_size=0.2,      # 20% validation
        random_state=42    # per riproducibilità
    )


    # Choose converter
    
    #converter = EventFrameConverter(height=128, width=128)
    #converter = StackedFrameConverter(128, 128, num_frames=5)
    #converter = TimeSurfaceConverter(128, 128, tau=50000)
    converter = VoxelGridConverter(128, 128, num_bins=5)
    
    dummy_event = {'x': np.array([0]), 'y': np.array([0]), 't': np.array([0]), 'p': np.array([1])}
    num_channels = converter.convert(dummy_event).shape[0]
    print(f"Using converter {converter.__class__.__name__} with {num_channels} channels")

    # Prepare datasets & loaders
    train_dataset = DVSGestureDataset(train_events, train_labels, converter, precompute=True)
    test_dataset = DVSGestureDataset(test_events, test_labels, converter, precompute=True)
    validation_dataset = DVSGestureDataset(validation_events, validation_labels, converter, precompute=True)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Build model
    model = DVSGestureCNN(num_input_channels=num_channels, num_classes=11)
    trainer = ModelTrainer(model)
    start_time = time.time()
    best_accuracy = trainer.train(train_loader, validation_loader, num_epochs=20, lr=0.001)
    end_time = time.time()
    time_elapsed = end_time - start_time
    print(f"Training + evaluation time: {time_elapsed:.2f} seconds")

    # Confusion matrix
    class_names = ["Hand Clapping", "Right Hand Wave", "Left Hand Wave",
                   "Right Arm CW", "Right Arm CCW", "Left Arm CW", "Left Arm CCW",
                   "Arm Roll", "Air Drums", "Air Guitar", "Other"]
    # trainer.plot_confusion_matrix(test_loader, class_names)

    print("Computing sparsity...")

    train_sparsity = compute_sparsity(train_loader)
    test_sparsity = compute_sparsity(test_loader)

    print(f"Train Sparsity: {train_sparsity*100:.2f}%")
    print(f"Test Sparsity: {test_sparsity*100:.2f}%")

    # Final Test Inference
    print("Loading best model for final test evaluation...")
    model.load_state_dict(torch.load('best_model.pth'))

    test_loss, test_acc, test_loss_std = trainer.evaluate(test_loader)

    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Test Loss Std Dev: {test_loss_std:.4f}")

    return best_accuracy, time_elapsed