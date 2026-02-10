import numpy as np

def structured_to_event_dict(events_struct):
    return {
        'x': events_struct['x'].astype(np.int32),
        'y': events_struct['y'].astype(np.int32),
        't': events_struct['t'].astype(np.int64),
        'p': events_struct['p'].astype(np.int8),
    }

def tuple_events_to_event_dict(filtered_events):
    """
    filtered_events: list of (x, y, t, p)
    """
    if len(filtered_events) == 0:
        return {
            'x': np.array([], dtype=np.int32),
            'y': np.array([], dtype=np.int32),
            't': np.array([], dtype=np.int64),
            'p': np.array([], dtype=np.int8),
        }

    filtered_events = np.array(filtered_events)

    return {
        'x': filtered_events[:, 0].astype(np.int32),
        'y': filtered_events[:, 1].astype(np.int32),
        't': filtered_events[:, 2].astype(np.int64),
        'p': filtered_events[:, 3].astype(np.int8),
    }
