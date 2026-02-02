from Filtering_techniques.OMSSaliencyMapFiltering import OMSFiltering
from Filtering_techniques.MaskAdaptiveElbow  import AdaptiveElbowOMSFiltering
from Filtering_techniques.MaskGoalOriented import MaskGoalOrientedOMSFiltering
from Filtering_techniques.MaskMeanStandardDeviation import MaskMeanStandardDeviation
from Filtering_techniques.MaskGlobalSaliencyBasedCropping import MaskGlobalSaliencyBasedCropping

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

    # Initialize and run Global Saliency Based Cropping
    GlobalSaliencyCropper = MaskGlobalSaliencyBasedCropping("DVSGesture")
    Use_percentile = True
    percentile = 90
    threshold = 0.4
    filred_events, OMS_norm, cropped_OMS_map, crop_box = GlobalSaliencyCropper.MaskGlobalSaliency_filtering(Use_percentile, percentile, threshold)
    GlobalSaliencyCropper.MaskGlobalSaliency_filtering_visualization(cropped_OMS_map, OMS_norm)

