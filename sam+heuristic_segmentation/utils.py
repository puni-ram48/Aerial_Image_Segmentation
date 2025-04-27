"""
Utility functions for resizing images, directory handling, and feature extraction.
"""

import os
import numpy as np
import cv2
from PIL import Image

def resize_image(image, target_size=(1024, 1024)):
    """
    Resize a PIL image to a target size.

    Args:
        image (PIL.Image): Input image.
        target_size (tuple): Target size (width, height).

    Returns:
        np.ndarray: Resized image as NumPy array.
    """
    return np.array(image.resize(target_size, Image.LANCZOS))

def ensure_dir(path):
    """
    Ensure a directory exists, create if it does not.

    Args:
        path (str): Directory path.
    """
    os.makedirs(path, exist_ok=True)

def get_hsv_mean(rgb_image, mask):
    """
    Compute mean HSV values inside a masked region.

    Args:
        rgb_image (np.ndarray): RGB image.
        mask (np.ndarray): Binary mask.

    Returns:
        tuple: Mean (H, S, V) values.
    """
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    region = hsv_image[mask > 0]
    if region.size == 0:
        return (0, 0, 0)
    return tuple(np.mean(region, axis=0))

def get_contour_features(mask):
    """
    Compute contour features like area, perimeter, and aspect ratio.

    Args:
        mask (np.ndarray): Binary mask.

    Returns:
        tuple: (area, perimeter, aspect_ratio)
    """
    mask_uint8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, 1
    max_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(max_contour)
    perimeter = cv2.arcLength(max_contour, True)
    x, y, w, h = cv2.boundingRect(max_contour)
    aspect_ratio = max(w, h) / max(1, min(w, h))
    return area, perimeter, aspect_ratio
