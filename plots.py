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

plt.figure(figsize=(14, 7))

offset_x = 0.10

for i in range(len(techniques)):
    plt.scatter(x[i], filter_ratio[i],
                s=160, color=colors[i], edgecolor='black')
    plt.scatter(x[i], test_accuracy[i],
                s=160, color=colors[i], edgecolor='black')

    plt.text(x[i] + offset_x, filter_ratio[i], "1",
             va='center', fontsize=14, fontweight='bold')
    
    plt.text(x[i] + offset_x, test_accuracy[i], "2",
             va='center', fontsize=14, fontweight='bold')


plt.axhline(y=baseline_accuracy, linestyle='--', linewidth=2)

plt.xticks(x, techniques, rotation=45, ha='right', fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel("1. Filtering Ratio   2. Test Accuracy", fontsize=16)

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


# -----------------------------------------
# Parameter Tuning Plot
# -----------------------------------------

# -----------------------------
# Data
# -----------------------------

data = {
    "OMS": {
        "thresholds": [0.1, 0.15, 0.2, 0.25, 0.3],
        "accuracy": [78.03, 72.35, 78.41, 75.76, 76.14],
        "time": [1198.67, 1189.20, 1184.13, 1164.51, 1157.18]
    },
    "Goal Oriented Thresholding": {
        "thresholds": [1, 2, 5, 10, 20, 30, 40],
        "accuracy": [49.24, 55.30, 66.67, 73.11, 79.55, 80.68, 79.55],
        "time": [2757.77, 2767.84, 2786.49, 2803.81, 2847.55, 2846.03, 2860.97]
    },
    "Mean Standard Deviation": {
        "thresholds": [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
        "accuracy": [77.65, 74.62, 74.62, 73.11, 68.56, 62.12, 56.44],
        "time": [2092.68, 2063.38, 2058.77, 2058.92, 2004.57, 2009.11, 1977.91]
    },
    "Global Saliency Crop (Percentile)": {
        "thresholds": [85, 88, 90, 92, 95],
        "accuracy": [82.20, 79.55, 71.59, 77.65, 76.14],
        "time": [2912.54, 2475.92, 2141.89, 2157.32, 2094.80]
    },
    "Global Saliency Crop (No Percentile)": {
        "thresholds": [0.15, 0.2, 0.25],
        "accuracy": [81.44, 78.03, 78.79],
        "time": [2121.86, 2129.05, 2084.48]
    }
}

# -----------------------------
# Plotting
# -----------------------------

for technique, values in data.items():
    thresholds = values["thresholds"]
    accuracy = values["accuracy"]
    time = values["time"]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Accuracy plot (light blue)
    line1 = ax1.plot(
        thresholds,
        accuracy,
        color="#3aacff",
        marker="o",
        linewidth=2.5,
        markersize=7,
        label="Test Accuracy (%)"
    )
    ax1.set_xlabel("Threshold", fontsize=12)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=12, color="#3aacff")
    ax1.tick_params(axis='y', labelcolor="#3aacff")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Time plot (light green)
    ax2 = ax1.twinx()
    line2 = ax2.plot(
        thresholds,
        time,
        color="#34d7ba",
        marker="s",
        linewidth=2.5,
        markersize=7,
        label="Filtering Time (s)"
    )
    ax2.set_ylabel("Filtering Time (seconds)", fontsize=12, color="#34d7ba")
    ax2.tick_params(axis='y', labelcolor="#34d7ba")

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best", fontsize=11)

    plt.title(f"{technique}\nAccuracy and Filtering Time vs Threshold", fontsize=14)
    plt.tight_layout()
    plt.show()