import tonic

from Filtering_techniques.OMSSaliencyMapFiltering import OMSFiltering
from Filtering_techniques.MaskAdaptiveElbow  import AdaptiveElbowOMSFiltering
from Filtering_techniques.MaskGoalOriented import MaskGoalOrientedOMSFiltering
from Filtering_techniques.MaskMeanStandardDeviation import MaskMeanStandardDeviation
from Filtering_techniques.MaskGlobalSaliencyBasedCropping import MaskGlobalSaliencyBasedCropping
from Filtering_techniques.Random_filtering import RandomEventFiltering
from Filtering_techniques.Denoise import Denoise
from Filtering_techniques.RandomCropFiltering import RandomCropFiltering
from Classification.ComplexCNN import train_model

if __name__ == "__main__":
    
    """
    # Initialize and run OMS Filtering
    OMSfilter = OMSFiltering("DVSGesture")
    OMSMap, I_filtered = OMSfilter.OMS_filtering()
    OMSfilter.OMS_filtering_visualization(OMSMap, I_filtered)
    """

    """
    # Initialize and run Adaptive Elbow Thresholding
    AdaptiveElbowFilter = AdaptiveElbowOMSFiltering("DVSGesture")
    filtered_events, masked_OMS, OMSMap = AdaptiveElbowFilter.Albowdaptive_thresholding()
    AdaptiveElbowFilter.AdaptiveElbow_filtering_visualization(OMSMap, masked_OMS)
    """
    """
    # Initialize and run Goal Oriented Thresholding
    GoalOrientedFilter = MaskGoalOrientedOMSFiltering("DVSGesture")
    keep_percent = 5  # Keep top 5%
    filtered_events, masked_OMS, OMSMap = GoalOrientedFilter.Goadaptive_thresholding(keep_percent) 
    GoalOrientedFilter.GoalOriented_filtering_visualization(OMSMap, masked_OMS)
    """
    """
    # Initialize and run Mean and Standard Deviation Thresholding
    MeanStdFilter = MaskMeanStandardDeviation("DVSGesture")
    k_sigma = 2.0  # Parameter for thresholding
    filtered_events = MeanStdFilter.Mean_std_thresholding(k_sigma)
    MeanStdFilter.MeanStd_filtering_visualization(filtered_events, k_sigma)
    """
    """
    # Initialize and run Global Saliency Based Cropping
    GlobalSaliencyCropper = MaskGlobalSaliencyBasedCropping("DVSGesture")
    Use_percentile = True
    percentile = 90
    threshold = 0.4
    filred_events, OMS_norm, cropped_OMS_map, crop_box = GlobalSaliencyCropper.MaskGlobalSaliency_filtering(Use_percentile, percentile, threshold)
    GlobalSaliencyCropper.MaskGlobalSaliency_filtering_visualization(cropped_OMS_map, OMS_norm)
    """
    """
    # Initialize and run Random Filtering 
    RandomFiltering = RandomEventFiltering("DVSGesture")
    random_keep_prob = 0.2  # Keep 30% of events randomly
    window_pos_random, xs_rand, filtered_events = RandomFiltering.Random_filtering(random_keep_prob)
    RandomFiltering.Random_filtering_visualization(window_pos_random, xs_rand)   
    """
    """
    # Initialize and run Denoising Filtering
    DenoiseFiltering = Denoise("DVSGesture")
    events_denoised = DenoiseFiltering.Denoise_filtering()
    DenoiseFiltering.Denoise_filtering_visualization()
    """

    """
    # Initialize and run Random Cropping Filtering
    sensor_size = (128, 128)      # original DVS resolution
    crop_size = (64, 64)         # desired crop size
    RandomCrop_filtering = RandomCropFiltering("DVSGesture", sensor_size, crop_size)
    events_cropped = RandomCrop_filtering.RandomCrop_filtering(sensor_size, crop_size)
    RandomCrop_filtering.RandomCrop_filtering_visualization(events_cropped)
    """

    # Training the ComplexCNN model
    print("Loading DVSGesture dataset...")
    dataset_training = tonic.datasets.DVSGesture(save_to="../Datasets", train=True)
    dataset_testing = tonic.datasets.DVSGesture(save_to="../Datasets", train=False)

    train_model(dataset_training, dataset_testing)


