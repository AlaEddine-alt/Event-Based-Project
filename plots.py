import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

# Data
techniques = [
    "OMS",
    "Attention",
    "Adaptive Elbow",
    "Goal Oriented",
    "Mean-StdDev",
    "Global Saliency Crop",
    "Denoising",
    "Random Crop Filtering"
]

test_accuracy = np.array([
    79.17, 72.35, 71.59, 80.68,
    77.27, 81.44, 83.71, 43.56
])

filter_ratio = np.array([
    27.42, 27.73, 80.53, 15.20,
    28.03, 9.13, 36.11, 59.36
])

baseline_accuracy = 81.44

x = np.arange(len(techniques))
colors = plt.cm.tab10(np.linspace(0, 1, len(techniques)))

plt.figure(figsize=(16, 7))

offset_x = 0.10

for i in range(len(techniques)):
    plt.scatter(x[i], filter_ratio[i],
                s=160, color=colors[i], edgecolor='black')
    plt.scatter(x[i], test_accuracy[i],
                s=160, color=colors[i], edgecolor='black')

    # Numeri più grandi
    plt.text(x[i] + offset_x, filter_ratio[i], "1",
             va='center', fontsize=14, fontweight='bold')
    
    plt.text(x[i] + offset_x, test_accuracy[i], "2",
             va='center', fontsize=14, fontweight='bold')

# Baseline più visibile
plt.axhline(y=baseline_accuracy, linestyle='--', linewidth=2)

# Font più grandi
plt.xticks(x, techniques, rotation=45, ha='right', fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel("1. Filtering Ratio   2. Test Accuracy", fontsize=16)

plt.ylim(0, 100)
plt.grid(True, axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()

# Data
techniques = [
    "OMS",
    "Attention",
    "Adaptive Elbow",
    "Goal Oriented",
    "Mean-StdDev",
    "Global Saliency Crop",
    "Denoising",
    "Random Crop Filtering"
]

test_accuracy = np.array([
    79.17,
    72.35,
    71.59,
    80.68,
    77.27,
    81.44,
    83.71,
    43.56
])

filter_ratio = np.array([
    27.42,
    27.73,
    80.53,
    15.20,
    28.03,
    9.13,
    36.11,
    59.36
])

baseline_accuracy = 81.44

x = np.arange(len(techniques))
colors = plt.cm.tab10(np.linspace(0, 1, len(techniques)))

plt.figure(figsize=(9, 6))

offset_x = 0.08  # horizontal shift for labels

for i in range(len(techniques)):
    # Scatter points
    plt.scatter(x[i], filter_ratio[i], 
                s=120, color=colors[i], edgecolor='black')
    plt.scatter(x[i], test_accuracy[i], 
                s=120, color=colors[i], edgecolor='black')

    # Shift labels slightly to the right
    plt.text(x[i] + offset_x, filter_ratio[i], "1",
             va='center', fontsize=9, fontweight='bold')
    
    plt.text(x[i] + offset_x, test_accuracy[i], "2",
             va='center', fontsize=9, fontweight='bold')

# Baseline horizontal line
plt.axhline(y=baseline_accuracy, linestyle='--', linewidth=1.5)

# Formatting
plt.xticks(x, techniques, rotation=45, ha='right')
plt.ylabel("1. Filtering Ratio   2. Test Accuracy")
plt.ylim(0, 100)
plt.grid(True, axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()

# Data
techniques = [
    "OMS Filtering",
    "Attention Filtering",
    "Adaptive Elbow Thresholding",
    "Goal Oriented Thresholding",
    "Mean-StdDev Thresholding",
    "Global Saliency Based Cropping",
    "Denoising Filtering",
    "Random Crop Filtering"
]

filter_ratio = [0.2742, 0.2773, 0.8053, 0.1520, 0.2803, 0.0913, 0.3611, 0.5936]
test_accuracy = [79.17, 72.35, 71.59, 80.68, 77.27, 81.44, 83.71, 43.56]
total_time = [1178.45, 2387.57, 2207.13, 2861.87, 2097.54, 2128.1, 2471.5, 42.25]

techniques = np.array([
    "OMS", "Attention", "Adaptive Elbow", "Goal Oriented",
    "Mean-StdDev", "Saliency Crop", "Denoising", "Random Crop"
])

filter_ratio = np.array([0.2742, 0.2773, 0.8053, 0.1520, 0.2803, 0.0913, 0.3611, 0.5936])
test_accuracy = np.array([79.17, 72.35, 71.59, 80.68, 77.27, 81.44, 83.71, 43.56])

# Sort by filter ratio
sorted_indices = np.argsort(filter_ratio)
sorted_ratio = filter_ratio[sorted_indices]
sorted_accuracy = test_accuracy[sorted_indices]
sorted_techniques = techniques[sorted_indices]

plt.figure()
plt.plot(sorted_ratio, sorted_accuracy, marker='o')

for i, txt in enumerate(sorted_techniques):
    plt.annotate(txt, (sorted_ratio[i], sorted_accuracy[i]))

plt.xlabel("Filter Ratio (Data Removed)")
plt.ylabel("Test Accuracy (%)")
plt.title("Performance Curve: Accuracy vs Filter Ratio")
plt.show()

# -----------------------------
# 1. Bar Chart – Test Accuracy
# -----------------------------
plt.figure()
plt.bar(techniques, test_accuracy)
plt.xticks(rotation=75)
plt.xlabel("Filtering Technique")
plt.ylabel("Test Accuracy (%)")
plt.title("Test Accuracy Comparison")
plt.tight_layout()
plt.show()

# -----------------------------
# 2. Bar Chart – Total Time
# -----------------------------
plt.figure()
plt.bar(techniques, total_time)
plt.xticks(rotation=75)
plt.xlabel("Filtering Technique")
plt.ylabel("Total Time (sec)")
plt.title("Total Time Comparison")
plt.tight_layout()
plt.show()

# -----------------------------------------
# 3. Scatter Plot – Filter Ratio vs Accuracy
# -----------------------------------------
plt.figure()
plt.scatter(filter_ratio, test_accuracy)

for i, txt in enumerate(techniques):
    plt.annotate(txt, (filter_ratio[i], test_accuracy[i]))

plt.xlabel("Filter Ratio (Data Removed)")
plt.ylabel("Test Accuracy (%)")
plt.title("Filter Ratio vs Test Accuracy")
plt.tight_layout()
plt.show()