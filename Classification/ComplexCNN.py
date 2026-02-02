import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import tonic
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time


class EventFrameConverter:
    """
    Converts events into a single frame (very fast, no temporal info)
    """
    def __init__(self, height=128, width=128):
        self.height = height
        self.width = width
    
    def convert(self, events):
        if len(events['x']) == 0:
            return torch.zeros((2, self.height, self.width), dtype=torch.float32)
        
        x = events['x'].astype(np.int32)
        y = events['y'].astype(np.int32)
        p = events['p'].astype(np.int32)
        
        # Filtra eventi fuori bounds
        valid = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        x, y, p = x[valid], y[valid], p[valid]
        
        # 2 canali: ON e OFF events
        frame = np.zeros((2, self.height, self.width), dtype=np.float32)
        np.add.at(frame[0], (y[p == 1], x[p == 1]), 1)  # ON events
        np.add.at(frame[1], (y[p == 0], x[p == 0]), 1)  # OFF events
        
        return torch.from_numpy(frame)


class StackedFrameConverter:
    """
    Devide events in N temporal frames (fast, simple temporal info)
    """
    def __init__(self, height=128, width=128, num_frames=5):
        self.height = height
        self.width = width
        self.num_frames = num_frames
    
    def convert(self, events):
        if len(events['x']) == 0:
            return torch.zeros((self.num_frames * 2, self.height, self.width), dtype=torch.float32)
        
        x = events['x'].astype(np.int32)
        y = events['y'].astype(np.int32)
        t = events['t']
        p = events['p'].astype(np.int32)
        
        # filters out of bounds events
        valid = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        x, y, t, p = x[valid], y[valid], t[valid], p[valid]
        
        # Devide in temporal chunks
        t_norm = (t - t.min()) / (t.max() - t.min() + 1e-6)
        t_idx = (t_norm * (self.num_frames - 1)).astype(np.int32)
        
        # create frames for each chunk (2 channels for ON/OFF)
        frames = np.zeros((self.num_frames * 2, self.height, self.width), dtype=np.float32)
        
        for i in range(self.num_frames):
            mask = t_idx == i
            if mask.any():
                x_i, y_i, p_i = x[mask], y[mask], p[mask]
                np.add.at(frames[i*2], (y_i[p_i == 1], x_i[p_i == 1]), 1)      # ON
                np.add.at(frames[i*2+1], (y_i[p_i == 0], x_i[p_i == 0]), 1)    # OFF
        
        return torch.from_numpy(frames)


class TimeSurfaceConverter:
    """
    Time surface: memorize timestamp of last event per pixel 
    """
    def __init__(self, height=128, width=128, tau=50000):  # tau in microseconds
        self.height = height
        self.width = width
        self.tau = tau
    
    def convert(self, events):
        if len(events['x']) == 0:
            return torch.zeros((2, self.height, self.width), dtype=torch.float32)
        
        x = events['x'].astype(np.int32)
        y = events['y'].astype(np.int32)
        t = events['t']
        p = events['p'].astype(np.int32)
        
        # Filters events out of bounds
        valid = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        x, y, t, p = x[valid], y[valid], t[valid], p[valid]
        
        # Time surface: exp decay basato su timestamp
        surface = np.zeros((2, self.height, self.width), dtype=np.float32)
        
        if len(t) > 0:
            t_max = t.max()
            for i in range(len(x)):
                channel = p[i]
                dt = t_max - t[i]
                surface[channel, y[i], x[i]] = np.exp(-dt / self.tau)
        
        return torch.from_numpy(surface)


class VoxelGridConverter:
    """
    Voxel grid ottimizzato (medium-fast, good temporal info)
    """
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
        
        t_norm = (t - t.min()) / (t.max() - t.min() + 1e-6)
        t_idx = (t_norm * (self.num_bins - 1)).astype(np.int32)
        
        valid = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        x, y, t_idx, p = x[valid], y[valid], t_idx[valid], p[valid]
        
        voxel = np.zeros((self.num_bins, self.height, self.width), dtype=np.float32)
        polarity_values = 2 * p - 1
        np.add.at(voxel, (t_idx, y, x), polarity_values)
        
        return torch.from_numpy(voxel)


class DVSGestureDataset(Dataset):
    def __init__(self, event_data, labels, converter, precompute=True):
        self.labels = labels
        self.converter = converter
        
        if precompute:
            print(f"Pre-computing {len(event_data)} representations...")
            self.data = []
            for i, events in enumerate(event_data):
                representation = self.converter.convert(events)
                self.data.append(representation)
                if (i + 1) % 100 == 0:
                    print(f"  Converted {i + 1}/{len(event_data)}")
            self.event_data = None
        else:
            self.event_data = event_data
            self.data = None
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        
        if self.data is not None:
            representation = self.data[idx]
        else:
            events = self.event_data[idx]
            representation = self.converter.convert(events)
        
        return representation, label


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class DVSGestureCNN(nn.Module):
    def __init__(self, num_input_channels, num_classes=11, dropout=0.5):
        super(DVSGestureCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=2, 
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
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


class ModelTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', stability_window=5):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.stability_window = stability_window  # number of epochs to compute stability

    def train_epoch(self, train_loader, optimizer):
        self.model.train()
        total_loss = 0
        batch_losses = []
        correct = 0
        total = 0

        for data, labels in train_loader:
            data = data.to(self.device)
            labels = labels.to(self.device)

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

        avg_loss = total_loss / len(train_loader)
        loss_std = np.std(batch_losses)  # measure training stability
        accuracy = 100. * correct / total
        return avg_loss, accuracy, loss_std

    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0
        batch_losses = []
        correct = 0
        total = 0

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                batch_losses.append(loss.item())

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(test_loader)
        loss_std = np.std(batch_losses)  # measure stability on test
        accuracy = 100. * correct / total
        return avg_loss, accuracy, loss_std

    def train(self, train_loader, test_loader, num_epochs=50, lr=0.001, convergence_threshold=95.0):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

        best_acc = 0
        history = {'train_loss': [], 'train_acc': [], 'train_loss_std': [],
                   'test_loss': [], 'test_acc': [], 'test_loss_std': []}
        convergence_epoch = None

        for epoch in range(num_epochs):
            train_loss, train_acc, train_loss_std = self.train_epoch(train_loader, optimizer)
            test_loss, test_acc, test_loss_std = self.evaluate(test_loader)
            scheduler.step()

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_loss_std'].append(train_loss_std)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            history['test_loss_std'].append(test_loss_std)

            print(f'Epoch {epoch+1}/{num_epochs}')
            print(f'Train Loss: {train_loss:.4f} ± {train_loss_std:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Test Loss: {test_loss:.4f} ± {test_loss_std:.4f}, Test Acc: {test_acc:.2f}%')
            print('-' * 60)

            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(self.model.state_dict(), 'best_model.pth')

            # Check convergence speed
            if convergence_epoch is None and test_acc >= convergence_threshold:
                convergence_epoch = epoch + 1

        print(f'Best Test Accuracy: {best_acc:.2f}%')
        if convergence_epoch:
            print(f"Converged (≥{convergence_threshold}% acc) at epoch {convergence_epoch}")
        else:
            print(f"Did not reach {convergence_threshold}% accuracy in {num_epochs} epochs.")

        return history


def compare_representations(train_events, train_labels, test_events, test_labels, 
                           num_epochs=5, batch_size=32):
    """
    Confront different event representations
    """
    HEIGHT = 128
    WIDTH = 128
    NUM_CLASSES = 11
    
    # Define rapresentations to test
    representations = {
        'EventFrame': (EventFrameConverter(HEIGHT, WIDTH), 2), # 2 channels: ON/OFF
        'StackedFrames_5': (StackedFrameConverter(HEIGHT, WIDTH, 5), 10), # 5 frames x 2 channels
        'TimeSurfaceConverter': (TimeSurfaceConverter(HEIGHT, WIDTH, tau=50000), 2), # 2 channels: ON/OFF
        'VoxelGrid_5': (VoxelGridConverter(HEIGHT, WIDTH, 5), 5), # 5 temporal bins
    }
    
    results = {}
    
    for name, (converter, num_channels) in representations.items():
        print(f"\n{'='*70}")
        print(f"Testing representation: {name}")
        print(f"{'='*70}\n")
        
        # Create dataset
        train_dataset = DVSGestureDataset(train_events, train_labels, converter, precompute=True)
        test_dataset = DVSGestureDataset(test_events, test_labels, converter, precompute=True)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Create model
        model = DVSGestureCNN(num_input_channels=num_channels, num_classes=NUM_CLASSES)
        print(f"Numero parametri: {sum(p.numel() for p in model.parameters()):,}")
        
        trainer = ModelTrainer(model)
        history = trainer.train(train_loader, test_loader, num_epochs=num_epochs, lr=0.001)
        
        results[name] = history
    
    return results

def plot_confusion_matrix(model, dataloader, class_names, device):
    """
    Computes and plots the confusion matrix for a trained model.

    Args:
        model: trained PyTorch model
        dataloader: DataLoader for the test set
        class_names: list of class labels (length = num_classes)
        device: 'cpu' or 'cuda'
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=True)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load DVSGesture dataset
    print("Loading DVSGesture dataset...")
    dataset_training = tonic.datasets.DVSGesture(save_to="../Datasets", train=True)
    dataset_testing = tonic.datasets.DVSGesture(save_to="../Datasets", train=False)
    
    print(f"Training samples: {len(dataset_training)}")
    print(f"Testing samples: {len(dataset_testing)}")
    
    # Extract events and labels
    train_events = []
    train_labels = []
    print("Loading training data...")
    for i in range(len(dataset_training)):
        events, label = dataset_training[i]
        events_dict = {
            'x': events['x'],
            'y': events['y'],
            't': events['t'],
            'p': events['p']
        }
        train_events.append(events_dict)
        train_labels.append(label)
        if (i + 1) % 100 == 0:
            print(f"  Loaded {i + 1}/{len(dataset_training)}")
    
    test_events = []
    test_labels = []
    print("Loading testing data...")
    for i in range(len(dataset_testing)):
        events, label = dataset_testing[i]
        events_dict = {
            'x': events['x'],
            'y': events['y'],
            't': events['t'],
            'p': events['p']
        }
        test_events.append(events_dict)
        test_labels.append(label)
        if (i + 1) % 100 == 0:
            print(f"  Loaded {i + 1}/{len(dataset_testing)}")
    
    
    # Option 1: Confront all representations (slower)
    # results = compare_representations(train_events, train_labels, test_events, test_labels, num_epochs=5, batch_size=32)
    
    # OPZIONE 2: Use only Event Frame representation (faster)
    converter = EventFrameConverter(height=128, width=128)
    train_dataset = DVSGestureDataset(train_events, train_labels, converter, precompute=True)
    test_dataset = DVSGestureDataset(test_events, test_labels, converter, precompute=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    model = DVSGestureCNN(num_input_channels=2, num_classes=11)
    trainer = ModelTrainer(model)
    start_time = time.time()
    history = trainer.train(train_loader, test_loader, num_epochs=20, lr=0.001)
    end_time = time.time()
    print(f"Training and evaluation time: {end_time - start_time:.2f} seconds")

    class_names = [
    "Hand Clapping", "Right Hand Wave", "Left Hand Wave",
    "Right Arm CW", "Right Arm CCW",
    "Left Arm CW", "Left Arm CCW",
    "Arm Roll", "Air Drums", "Air Guitar", "Other"
    ]

    plot_confusion_matrix(
        model=model,
        dataloader=test_loader,
        class_names=class_names,
        device=trainer.device
    )