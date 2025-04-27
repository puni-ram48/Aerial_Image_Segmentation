"""
Evaluation script to compute per-image Pixel Accuracy, and per-class metrics (IoU, Dice, Precision, Recall, F1).
"""

import os
import numpy as np
import cv2
from sklearn.metrics import precision_score, recall_score, f1_score
from config import config


def parse_label_file(label_file_path):
    """
    Parse label file and create mappings.

    Args:
        label_file_path (str): Path to label file.

    Returns:
        tuple: (class_mapping, color_to_class)
    """
    CLASS_MAPPING = {}
    COLOR_TO_CLASS = {}
    with open(label_file_path, "r") as f:
        lines = f.readlines()
    class_id = 0
    for line in lines:
        if line.startswith("#") or line.strip() == "":
            continue
        parts = line.strip().split(":")
        class_name = parts[0]
        rgb_values = tuple(map(int, parts[1].split(",")))
        CLASS_MAPPING[class_id] = class_name
        COLOR_TO_CLASS[rgb_values] = class_id
        class_id += 1
    return CLASS_MAPPING, COLOR_TO_CLASS

def convert_color_mask_to_class_ids(mask_path, color_map):
    """
    Convert color mask to class ID mask.

    Args:
        mask_path (str): Path to color mask.
        color_map (dict): Mapping from color to class ID.

    Returns:
        np.ndarray: Class ID mask.
    """
    mask = cv2.imread(mask_path)
    if mask is None:
        raise FileNotFoundError(f"Could not read {mask_path}")
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    class_mask = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
    for rgb_color, class_id in color_map.items():
        mask_pixels = np.all(mask_rgb == np.array(rgb_color), axis=-1)
        class_mask[mask_pixels] = class_id
    return class_mask

def compute_per_class_metrics(gt_mask, pred_mask, class_id):
    """
    Compute IoU, Dice, Precision, Recall, and F1 Score for a specific class.

    Args:
        gt_mask (np.ndarray): Ground truth class mask.
        pred_mask (np.ndarray): Predicted class mask.
        class_id (int): Class ID.

    Returns:
        tuple: (IoU, Dice, Precision, Recall, F1)
    """
    gt = (gt_mask == class_id)
    pred = (pred_mask == class_id)

    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()

    iou = intersection / (union + 1e-6)
    dice = (2 * intersection) / (gt.sum() + pred.sum() + 1e-6)

    precision = precision_score(gt.flatten(), pred.flatten(), average='binary', zero_division=0)
    recall = recall_score(gt.flatten(), pred.flatten(), average='binary', zero_division=0)
    f1 = f1_score(gt.flatten(), pred.flatten(), average='binary', zero_division=0)

    return iou, dice, precision, recall, f1

def evaluate_segmentation():
    """
    Full evaluation loop over all images with per-class and per-image metrics.
    """
    gt_dir = config["gt_dir"]
    pred_dir = config["output_dir"]
    label_file_path = config["label_file"]

    CLASS_MAPPING, COLOR_TO_CLASS = parse_label_file(label_file_path)

    gt_images = sorted(os.listdir(gt_dir))
    pred_images = sorted(os.listdir(pred_dir))

    per_class_metrics = {}  # {class_name: list of [iou, dice, precision, recall, f1]}
    for class_id, class_name in CLASS_MAPPING.items():
        if class_id == 0:
            continue  # Skip background
        per_class_metrics[class_name] = []

    pixel_accuracies = []  # List of pixel accuracies per image

    print("\n=== Image-wise Evaluation ===\n")

    for img_name in gt_images:
        gt_path = os.path.join(gt_dir, img_name)
        pred_path = os.path.join(pred_dir, img_name.replace(".png", ".png_classified.png"))

        if not os.path.exists(pred_path):
            print(f"Skipping {img_name}: predicted mask not found.")
            continue

        gt_mask = convert_color_mask_to_class_ids(gt_path, COLOR_TO_CLASS)
        pred_mask = convert_color_mask_to_class_ids(pred_path, COLOR_TO_CLASS)

        # Compute pixel accuracy
        pixel_accuracy = (gt_mask == pred_mask).sum() / gt_mask.size
        pixel_accuracies.append(pixel_accuracy)

        print(f"\nImage: {img_name}")
        print(f"  Pixel Accuracy: {pixel_accuracy:.4f}")

        # Compute per-class metrics
        for class_id, class_name in CLASS_MAPPING.items():
            if class_id == 0:
                continue  # Skip background

            iou, dice, precision, recall, f1 = compute_per_class_metrics(gt_mask, pred_mask, class_id)
            per_class_metrics[class_name].append([iou, dice, precision, recall, f1])

            print(f"  Class: {class_name}")
            print(f"    IoU: {iou:.4f}")
            print(f"    Dice Score: {dice:.4f}")
            print(f"    Precision: {precision:.4f}")
            print(f"    Recall: {recall:.4f}")
            print(f"    F1 Score: {f1:.4f}")

    # Summary
    print("\n=== Overall Evaluation Summary ===\n")

    # Mean Pixel Accuracy
    if pixel_accuracies:
        mean_pixel_accuracy = np.mean(pixel_accuracies)
        print(f"Mean Pixel Accuracy over all images: {mean_pixel_accuracy:.4f}")

    print("\n=== Overall Per-Class Metrics ===\n")
    for class_name, metrics_list in per_class_metrics.items():
        if len(metrics_list) == 0:
            print(f"Class {class_name}: No samples found.")
            continue
        metrics_array = np.array(metrics_list)
        mean_metrics = np.mean(metrics_array, axis=0)
        print(f"Class: {class_name}")
        print(f"  Mean IoU: {mean_metrics[0]:.4f}")
        print(f"  Mean Dice Score: {mean_metrics[1]:.4f}")
        print(f"  Mean Precision: {mean_metrics[2]:.4f}")
        print(f"  Mean Recall: {mean_metrics[3]:.4f}")
        print(f"  Mean F1 Score: {mean_metrics[4]:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    evaluate_segmentation()