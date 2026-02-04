import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tonic.datasets import DVSGesture
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 1. Custom Dataset
# ----------------------------
class DVSGestureFrames(Dataset):
    def __init__(self, train=True):
        # Setting download=True helps ensure files exist
        self.dataset = DVSGesture(save_to="C:/Datasets", train=train)
        self.height = 128
        self.width = 128

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            events, target = self.dataset[idx]
        except ValueError:
            # If a file is corrupted, load the first item instead to avoid crashing
            print(f"Warning: Corrupted file at index {idx}. Skipping...")
            events, target = self.dataset[0]
            
        xs = events["x"].astype(int)
        ys = events["y"].astype(int)
        pols = events["p"].astype(int)

        frame = np.zeros((2, self.height, self.width), dtype=np.float32)
        np.add.at(frame[1], (ys[pols==1], xs[pols==1]), 1.0)
        np.add.at(frame[0], (ys[pols==0], xs[pols==0]), 1.0)

        if frame.max() > 0:
            frame /= frame.max()

        return torch.tensor(frame, dtype=torch.float32), int(target)

# ----------------------------
# 2. CNN Model
# ----------------------------

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=11):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 128 -> 64 -> 32 -> 16 spatial size
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))  
        x = self.pool(F.relu(self.conv3(x)))  
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ----------------------------
# 3. Metric Calculations (Manual)
# ----------------------------
def calculate_metrics(y_true, y_pred, num_classes=11):
    acc = np.mean(y_true == y_pred)
    
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
        
    f1_scores = []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        f1_scores.append(f1)
    
    return acc, np.mean(f1_scores), cm

# ----------------------------
# 4. Main Loop
# ----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # Initialize Datasets
    full_dataset = DVSGestureFrames(train=True)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # FIX: Set num_workers=0 for stability on Windows CPU
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(DVSGestureFrames(train=False), batch_size=16, shuffle=False, num_workers=0)

    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("Starting training...")
    for epoch in range(20):
        model.train()
        epoch_loss = 0
        for frames, targets in train_loader:
            frames, targets = frames.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/20], Loss: {epoch_loss/len(train_loader):.4f}")

    # Test Phase
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for frames, targets in test_loader:
            frames = frames.to(device)
            outputs = model(frames)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    # Metrics
    y_true, y_pred = np.array(all_targets), np.array(all_preds)
    test_acc, test_f1, test_cm = calculate_metrics(y_true, y_pred)

    print(f"\nResults:\nAccuracy: {test_acc:.4f}\nF1-Macro: {test_f1:.4f}")
    
    # Simple Plot
    
    plt.figure(figsize=(8,6))
    plt.imshow(test_cm, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.show()