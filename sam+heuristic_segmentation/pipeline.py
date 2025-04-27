"""
Pipeline script to load SAM model, segment images, classify regions, and save outputs.
"""

import os
import numpy as np
import torch
from PIL import Image
import logging
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from utils import resize_image, ensure_dir, get_hsv_mean, get_contour_features
from config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define color codes for classes
ROAD_COLOR = [128, 64, 128]
GREEN_AREA_COLOR = [107, 142, 35]
BUILDING_COLOR = [70, 70, 70]
UNKNOWN_COLOR = [0, 0, 0]

def load_sam_model(checkpoint_path, device):
    """
    Load the SAM model from a checkpoint.

    Args:
        checkpoint_path (str): Path to model checkpoint.
        device (str): Device type.

    Returns:
        SamAutomaticMaskGenerator: Mask generator.
    """
    sam_model = sam_model_registry["vit_l"]()
    sam_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    sam_model = sam_model.to(device)
    return SamAutomaticMaskGenerator(
        model=sam_model,
        points_per_side=64,
        pred_iou_thresh=0.90,
        stability_score_thresh=0.7
    )

def classify_object(rgb_image, mask):
    """
    Classify the object based on color and shape properties.

    Args:
        rgb_image (np.ndarray): RGB image.
        mask (np.ndarray): Binary mask.

    Returns:
        str: Class label.
    """
    h, s, v = get_hsv_mean(rgb_image, mask)
    area, _, ar = get_contour_features(mask)

    if s < 50 and 70 <= v <= 210 and 1000 < area < 1e6:
        return "road"
    elif (20 <= h <= 100) and (s > 10) and (40 < v < 230) and (area > 300):
        return "green_area"
    elif area > 5000 and ar < 2.5 and not (20 <= h <= 110):
        return "building"
    else:
        return "unknown"

def classify_and_colorize(rgb_image, masks):
    """
    Apply classification and colorization to segmentation masks.

    Args:
        rgb_image (np.ndarray): Input RGB image.
        masks (list): List of masks.

    Returns:
        np.ndarray: Colorized classified output image.
    """
    output_image = rgb_image.copy()
    for mask_data in masks:
        seg = mask_data["segmentation"]
        cls = classify_object(rgb_image, seg)
        color = {
            "road": ROAD_COLOR,
            "green_area": GREEN_AREA_COLOR,
            "building": BUILDING_COLOR,
            "unknown": UNKNOWN_COLOR
        }.get(cls, UNKNOWN_COLOR)
        output_image[seg > 0] = color
    return output_image

def process_images():
    """
    Full processing pipeline to segment, classify and save images.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    checkpoint_path = "./saved_models/sam_vit_l.pth"
    mask_generator = load_sam_model(checkpoint_path, device)

    ensure_dir(config["output_dir"])

    for file_name in os.listdir(config["input_dir"]):
        if not file_name.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
            continue

        try:
            image_path = os.path.join(config["input_dir"], file_name)
            image_pil = Image.open(image_path).convert("RGB")
            image_np = resize_image(image_pil)

            masks = mask_generator.generate(image_np)
            final_image = classify_and_colorize(image_np, masks)

            output_path = os.path.join(config["output_dir"], f"{file_name}_classified.png")
            Image.fromarray(final_image).save(output_path)

            logging.info(f"Processed and saved: {output_path}")
        except Exception as e:
            logging.error(f"Failed processing {file_name}: {e}")

if __name__ == "__main__":
    process_images()
