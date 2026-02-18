
import os
import tonic
from functions.loadDatasetFunctions import load_events, DVSGestureNPYDataset
from Filtering_techniques.OMSSaliencyMapFiltering import OMSFiltering
from Filtering_techniques.AttentionMapFiltering import AttentionFiltering

import time
# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":

    print("Loading DVS Gesture dataset...")
    training_ROOT = "C:/Users/giuli/Desktop/Giulia/PER/Event-Based-Project/Datasets/DVSGesture/ibmGestureTrain"
    testing_ROOT = "C:/Users/giuli/Desktop/Giulia/PER/Event-Based-Project/Datasets/DVSGesture/ibmGestureTest"
    training_users = os.listdir(training_ROOT)
    test_users = os.listdir(testing_ROOT)

    train_dataset_raw = DVSGestureNPYDataset(training_ROOT, users=training_users)
    test_dataset_raw = DVSGestureNPYDataset(testing_ROOT, users=test_users)

    scale_factor = 3

    # ---------------------------
    # Pick one sample from training dataset
    # ---------------------------
    sample_idx = 0  # Change index to visualize different gestures
    events, label = train_dataset_raw[sample_idx]
    print(f"Loaded sample {sample_idx} with label: {label}")

    dataset = tonic.datasets.dvsgesture.DVSGesture(save_to = "../Datasets", train=True)
    events, label = dataset[sample_idx]

    # ---------------------------
    # Initialize OMSFiltering
    # ---------------------------
    oms_filter = OMSFiltering(
        event=events,
        scale_factor=scale_factor, 
        threshold=0.1
    )

    # ---------------------------
    # Run OMS + Attention filtering
    # ---------------------------
    start = time.time()
    OMS_map, events_dict, filtered_img, ERR = oms_filter.OMS_filtering()
    end = time.time()

    print("----- Filtering Stats -----")
    print(f"Filtered ratio (ERR): {ERR:.4f}")
    print(f"Processing time: {end - start:.4f}s")

    # ---------------------------
    # Visualize results
    # ---------------------------
    oms_filter.OMS_filtering_visualization(OMS_map, filtered_img)


    Attentionfilter = AttentionFiltering(events, scale_factor)
    filtered_event_Attention, saliency_map, Err_Attention = Attentionfilter.Attention_filtering()
    Attentionfilter.Attention_visualization(saliency_map)