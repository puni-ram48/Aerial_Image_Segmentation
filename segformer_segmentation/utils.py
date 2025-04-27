"""
utils.py

Helper functions for segmentation project.
Includes directory management, label file parsing, config loading, and mask conversion.
"""

import os
import json
import numpy as np
from PIL import Image

def ensure_dir(directory):
    """Ensures that a directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def parse_label_file(label_file_path):
    """
    Parses a label file mapping class names to RGB values.

    Args:
        label_file_path (str): Path to label file.

    Returns:
        tuple: (CLASS_MAPPING {id:name}, COLOR_TO_CLASS {(r,g,b): id})
    """
    CLASS_MAPPING = {}
    COLOR_TO_CLASS = {}

    with open(label_file_path, "r") as f:
        lines = f.readlines()

    class_id = 0
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.strip().split(":")
        class_name = parts[0]
        rgb = tuple(map(int, parts[1].split(",")))

        CLASS_MAPPING[class_id] = class_name
        COLOR_TO_CLASS[rgb] = class_id
        class_id += 1

    return CLASS_MAPPING, COLOR_TO_CLASS

def convert_color_mask_to_class_ids(mask, color_map):
    """
    Converts RGB mask to a class ID mask.

    Args:
        mask (np.ndarray): RGB mask.
        color_map (dict): RGB to class ID mapping.

    Returns:
        np.ndarray: Class ID mask.
    """
    height, width, _ = mask.shape
    class_mask = np.zeros((height, width), dtype=np.uint8)

    for rgb, class_id in color_map.items():
        matches = np.all(mask == np.array(rgb), axis=-1)
        class_mask[matches] = class_id

    return class_mask

def load_config(config_path):
    """
    Load configuration from a JSON file.

    Args:
        config_path (str): Path to config file.

    Returns:
        dict: Loaded configuration as a dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config
