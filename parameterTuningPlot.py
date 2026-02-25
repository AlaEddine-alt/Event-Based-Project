import matplotlib.pyplot as plt

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