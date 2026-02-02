from Filtering_techniques.OMSSaliencyMapFiltering import OMSFiltering
from Filtering_techniques.MaskAdaptiveElbow  import AdaptiveElbowOMSFiltering
from Filtering_techniques.MaskGoalOriented import MaskGoalOrientedOMSFiltering

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

    # Initialize and run Goal Oriented Thresholding
    GoalOrientedFilter = MaskGoalOrientedOMSFiltering("DVSGesture")
    keep_percent = 5  # Keep top 5%
    filtered_events, masked_OMS, OMSMap = GoalOrientedFilter.Goadaptive_thresholding(keep_percent) 
    GoalOrientedFilter.GoalOriented_filtering_visualization(OMSMap, masked_OMS)
