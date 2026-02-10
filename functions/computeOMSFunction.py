from functions.OMS_helpers import *

def compute_OMS(window_pos, net_center, net_surround, config):
    OMSpos = torch.tensor(window_pos, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)

    OMSpos_map, indexes_pos = egomotion(OMSpos, net_center, net_surround, config.DEVICE, config.MAX_Y, config.MAX_X,
                                        config.OMS_PARAMS['threshold'])

    OMSpos_map = OMSpos_map.squeeze(0).squeeze(0).cpu().detach().numpy()

    # print("OMS map stats:", OMSpos_map.min(), OMSpos_map.max(), OMSpos_map.mean())
    
    return OMSpos_map, indexes_pos