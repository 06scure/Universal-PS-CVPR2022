import cv2
import numpy as np

def make_error_heatmap(error_map):
    error = error_map[0, 0]
    valid_mask = error > 0

    error_uint8 = np.zeros_like(error, dtype=np.uint8)
    error_uint8[valid_mask] = np.clip(error[valid_mask] / 90.0 * 255.0, 0, 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(error_uint8, cv2.COLORMAP_JET)
    heatmap[~valid_mask] = 0
    return heatmap