import cv2
import numpy as np

def draw_graph_with_dots(events, suppressed, dropped, width=640, height=480):
    graph_img = np.ones((height, width, 3), dtype=np.uint8) * 255

    max_events = max(events + suppressed + dropped, default=1)
    margin = 50
    scale_x = (width - 2 * margin) / len(events) if events else 1
    scale_y = (height - 2 * margin) / max_events

    cv2.line(graph_img, (margin, height - margin), (width - margin, height - margin), (0,0,0), 2)
    cv2.line(graph_img, (margin, height - margin), (margin, margin), (0,0,0), 2)

    for i in range(len(events)):
        x = margin + int(i * scale_x)
        y_events = height - margin - int(events[i]*scale_y)
        cv2.circle(graph_img, (x, y_events), 4, (0,0,255), -1)
        y_suppressed = height - margin - int(suppressed[i]*scale_y)
        cv2.circle(graph_img, (x, y_suppressed), 4, (255,0,0), -1)
    return graph_img

def convert_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image