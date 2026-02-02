from Filtering_techniques.OMSSaliencyMapFiltering import OMSFiltering

if __name__ == "__main__":
    
    # Initialize and run OMS Filtering
    OMSfilter = OMSFiltering("DVSGesture")
    OMSMap, I_filtered = OMSfilter.OMS_filtering()
    OMSfilter.OMS_filtering_visualization(OMSMap, I_filtered)

    
