# Event-Based Project - Execution Instructions

## Table of Contents
1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Installation & Setup](#installation--setup)
4. [Project Structure](#project-structure)
5. [Running the Code](#running-the-code)
6. [Filtering Techniques](#filtering-techniques)
7. [Output and Results](#output-and-results)
8. [Troubleshooting](#troubleshooting)

---

## Project Overview

This project implements various filtering and event reduction techniques on **DVS (Dynamic Vision Sensor) Gesture** datasets. The pipeline processes event-based camera data using multiple saliency and attention-based filtering methods to reduce noise and improve classification efficiency.

**Key Features:**
- Multiple filtering techniques for event data
- Event Reduction Ratio (ERR) analysis
- CNN-based classification on filtered data
- Support for tonic library DVS datasets
- Performance metrics tracking

---

## Prerequisites

### System Requirements
- **Python 3.8+** (Python 3.9+ recommended)
- **Windows 10/11** (code configured for Windows paths)
- **RAM:** 8GB minimum (16GB+ recommended for full dataset processing)
- **Disk Space:** 20GB+ for datasets and filtered outputs

### Required Software
- pip (Python package manager)
- Git (optional, for version control)

---

## Installation & Setup

### Step 1: Create a Virtual Environment

Open **PowerShell** or **Command Prompt** and navigate to the project directory.

Create a virtual environment:

```powershell
python -m venv venv
```

### Step 2: Activate the Virtual Environment

On **Windows PowerShell**:

```powershell
.\venv\Scripts\Activate.ps1
```

On **Command Prompt**:

```cmd
.\venv\Scripts\activate.bat
```

You should see `(venv)` at the beginning of your terminal line.

### Step 3: Install Dependencies

Install all required packages using the requirements file:

```powershell
pip install -r requirements.txt
```

**Note:** If you encounter issues with OpenCV installation, run:

```powershell
python -m pip install numpy==1.26.4 opencv-python
```

### Step 4: Verify Installation

Test that all packages are installed correctly:

```powershell
python -c "import tonic; import torch; print('Installation successful!')"
```

---

## Project Structure

```
Event-Based-Project/
├── main_DVSGesture.py              # Main execution script (all filtering techniques)
├── classificationDVS.py             # Classification utilities
├── parameterTuning.py               # Parameter tuning script
├── plots.py                         # Plotting utilities
├── demo.py                          # geenrate images to compare techniques
│
├── Classification/                  # CNN models
│   ├── ComplexCNN.py               # Complex CNN for classification
│   └── SimpleCNN.py                # Simple CNN (baseline)
│
├── Filtering_techniques/            # All filtering implementations
│   ├── OMSSaliencyMapFiltering.py
│   ├── AttentionMapFiltering.py
│   ├── MaskAdaptiveElbow.py
│   ├── MaskGoalOriented.py
│   ├── MaskMeanStandardDeviation.py
│   ├── MaskGlobalSaliencyBasedCropping.py
│   ├── Denoise.py
│   └── RandomCropFiltering.py
│
├── functions/                       # Utility functions
│   ├── loadDatasetFunctions.py
│   ├── saveAndLoadFilteredData.py
│   ├── writeResultsFunctions.py
│   ├── computeOMSFunction.py
│   ├── OMS_helpers.py
│   ├── Speck_helpers.py
│   ├── attention_helpers.py
│   ├── visualizationFunctions.py
│   └── MainSpeckOMSAttention.py
│
└── requirements.txt                 # Python dependencies
```

---

## Running the Code

### Run Parameter Tuning

Optimize filtering hyperparameters for best performance:

```powershell
python parameterTuning.py
```

It's applied to filtering techniques OMS, Goal Oriented Thresholding, Mean Standard Deviation Thresholding, Global Saliency-based Cropping.

This generates hyperparameter tuning results and saves them to `tuning_results.txt`. 
Before running the main script, insert the hyperparameters that give the best accuracy. 

### Run All Filtering Techniques

Execute the main script that applies all filtering techniques and analyzes their performance:

```powershell
python main_DVSGesture.py
```

**What this script does:**
1. Loads the DVS Gesture dataset using tonic library
2. Applies 8 different filtering techniques to both training and test data:
   - OMS Filtering
   - Attention Filtering
   - Adaptive Elbow Thresholding
   - Goal-Oriented Thresholding
   - Mean and Standard Deviation Thresholding
   - Global Saliency-Based Cropping
   - Denoising
   - Random Cropping
3. Saves filtered datasets to `Datasets/FilteredDatasets/`
4. Computes Event Reduction Ratio (ERR) for each technique
5. Measures execution time for each filtering method
6. Writes results to `filtering_results.txt`

**Expected Output:**
```
Loading DVSGesture dataset...
OMS filtering
Average OMS Event Reduction Ratio (ERR) across all events: 0.5234
time OMS filtering = 324.53 seconds
[Additional filtering technique results...]
```

**Runtime:** Approximately 30-60 minutes depending on dataset size and hardware.


### Generate Visual Comparison of Filtering Techniques

Compare all 8 filtering techniques visually on individual samples:

```powershell
python demo.py
```

**What this script does:**
1. Loads the DVS Gesture training dataset
2. Processes the **first 100 samples** through all 8 filtering techniques in parallel
3. For each sample, generates a comprehensive comparison visualization:
   - **Left panel:** Raw event map (reference)
   - **Right panel (2×4 grid):** All filtering techniques side-by-side
     - OMS Filtering
     - Attention Filtering
     - Adaptive Elbow Thresholding
     - Goal-Oriented Thresholding
     - Mean and Standard Deviation Thresholding
     - Global Saliency-Based Cropping
     - Denoising
     - Random Cropping
4. Saves high-resolution comparison images to `ev_demo_results/` directory
5. Displays performance metrics (ERR) for each technique

**Output:** Individual PNG files for each sample showing side-by-side comparison

**Example filename:** `ev_demo_results/sample_001.png`, `sample_002.png`, etc.

**Runtime:** Approximately 10-20 minutes (faster than main script since it processes only 100 samples)

**Use Case:** 
- Quick visual inspection of how each filtering technique affects event data
- Parameter tuning validation before running full dataset processing
- Presentation and report generation

---

### Option 4: Visualize Results

Create plots from already existent filtering results:

```powershell
python plots.py
```

---

## Filtering Techniques

All filtering techniques are implemented in the `Filtering_techniques/` directory:

| Technique | File | Description | Key Parameter |
|-----------|------|-------------|----------------|
| **OMS Filtering** | OMSSaliencyMapFiltering.py | Orientation Map Saliency-based filtering | threshold = 0.2 |
| **Attention Filtering** | AttentionMapFiltering.py | Attention mechanism-based filtering | - |
| **Adaptive Elbow** | MaskAdaptiveElbow.py | Elbow detection for threshold selection | threshold_OMS |
| **Goal-Oriented** | MaskGoalOriented.py | Keep top percentage of important events | keep_percent = 30 |
| **Mean-StdDev** | MaskMeanStandardDeviation.py | Statistical threshold-based filtering | k_sigma = 0.75 |
| **Global Saliency Crop** | MaskGlobalSaliencyBasedCropping.py | Crop based on global saliency percentile | percentile = 85, threshold = 0.15 |
| **Denoising** | Denoise.py | Remove noise from event streams | - |
| **Random Crop** | RandomCropFiltering.py | Random spatial cropping | crop_size = (64, 64) |

### Key Parameters to Adjust

Open `main_DVSGesture.py` and modify these parameters to customize the filtering:

```python
scale_factor = 3                    # Downsampling factor for events
threshold_OMS = 0.2                 # OMS filtering threshold
keep_percent = 30                   # Goal-oriented keep percentage
k_sigma = 0.75                      # Mean-StdDev multiplier
percentile = 85                     # Saliency percentile threshold
threshold = 0.15                    # Global saliency threshold
sensor_size = (128, 128)            # Original DVS resolution
crop_size = (64, 64)                # Random crop size
```

---

## Output and Results

### Filtered Datasets

After running `main_DVSGesture.py`, filtered datasets are saved to:

```
Datasets/FilteredDatasets/
├── OMS/
├── Attention/
├── AdaptiveElbow/
├── GoalOrientedThresholding/
├── MeanStd/
├── GlobalSaliencyBasedCropping/
├── Denoise/
└── RandomCrop/
(each containing train/ and test/ subdirectories)
```

### Results Files

- **filtering_results.txt:** Event Reduction Ratio and execution time for each technique
- **results.txt:** Classification accuracy and overall performance metrics
- **tuning_results.txt:** Parameter tuning results (if parameterTuning.py is run)

### Metrics Tracked

For each filtering technique, the code computes:

| Metric | Description |
|--------|-------------|
| **ERR** | Event Reduction Ratio (percentage of events removed) |
| **Execution Time** | Time required to process entire dataset |
| **Accuracy** | Classification accuracy on filtered data (if CNN training is enabled) |

---

## Troubleshooting

### Issue: Module not found errors

**Solution:** Ensure virtual environment is activated:
```powershell
.\venv\Scripts\Activate.ps1
```

### Issue: Dataset download fails

**Solution:** The tonic library automatically downloads datasets. If it fails:
1. Check internet connection
2. Ensure sufficient disk space (>10GB)
3. Manually specify dataset path in the code:
```python
train_dataset_raw = tonic.datasets.dvsgesture.DVSGesture(
    save_to="[Insert specific directory here]", 
    train=True
)
```


