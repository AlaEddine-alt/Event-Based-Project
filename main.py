from Filtering_techniques.OMSSaliencyMapFiltering import OMSFiltering
from Filtering_techniques.MaskAdaptiveElbow  import AdaptiveElbowOMSFiltering

if __name__ == "__main__":
    
    """
    # Initialize and run OMS Filtering
    OMSfilter = OMSFiltering("DVSGesture")
    OMSMap, I_filtered = OMSfilter.OMS_filtering()
    OMSfilter.OMS_filtering_visualization(OMSMap, I_filtered)
    """

    
    # Initialize and run Adaptive Elbow Thresholding
    AdaptiveElbowFilter = AdaptiveElbowOMSFiltering("DVSGesture")
    filtered_events, masked_OMS, OMSMap = AdaptiveElbowFilter.Albowdaptive_thresholding()
    AdaptiveElbowFilter.AdaptiveElbow_filtering_visualization(OMSMap, masked_OMS)
