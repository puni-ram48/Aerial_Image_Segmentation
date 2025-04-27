"""
evaluation.py

Evaluate segmentation predictions using metrics like:
- Intersection over Union (IoU)
- Dice Score
- Pixel Accuracy
- Precision, Recall, F1 Score
- Class-wise and Image-wise evaluation

Results are printed and averaged across the dataset.
"""

import os
import argparse
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from PIL import Image
import utils

def compute_metrics(gt_mask, pred_mask, num_classes, class_mapping):
    """
    Computes IoU, Dice, Pixel Accuracy, Precision, Recall, and F1 Score for each class.

    Args:
        gt_mask (np.ndarray): Ground truth class ID mask.
        pred_mask (np.ndarray): Predicted class ID mask.
        num_classes (int): Number of classes.
        class_mapping (dict): Mapping from class ID to class name.

    Returns:
        dicts of metrics per class, and overall pixel accuracy.
    """
    iou = {}
    dice = {}

    for class_id in range(num_classes):
        gt = (gt_mask == class_id)
        pred = (pred_mask == class_id)

        intersection = np.logical_and(gt, pred).sum()
        union = np.logical_or(gt, pred).sum()

        iou[class_mapping[class_id]] = intersection / union if union else 0.0
        dice[class_mapping[class_id]] = (2 * intersection) / (gt.sum() + pred.sum() + 1e-6) if (gt.sum() + pred.sum()) else 0.0

    pixel_acc = (gt_mask == pred_mask).sum() / gt_mask.size

    precision = precision_score(gt_mask.flatten(), pred_mask.flatten(), average=None, labels=list(range(num_classes)), zero_division=0)
    recall = recall_score(gt_mask.flatten(), pred_mask.flatten(), average=None, labels=list(range(num_classes)), zero_division=0)
    f1 = f1_score(gt_mask.flatten(), pred_mask.flatten(), average=None, labels=list(range(num_classes)), zero_division=0)

    prf = {class_mapping[i]: {
            "Precision": precision[i],
            "Recall": recall[i],
            "F1": f1[i]
           } for i in range(num_classes)}

    return iou, dice, pixel_acc, prf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Segmentation Results")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    args = parser.parse_args()

    config = utils.load_config(args.config)
    pred_dir = config.get("output_dir")
    gt_dir = config.get("gt_dir")
    label_file = config.get("label_file")

    class_mapping, color_to_class = utils.parse_label_file(label_file)
    num_classes = len(class_mapping)

    pred_files = sorted(os.listdir(pred_dir))
    gt_files = sorted(os.listdir(gt_dir))
    matching = list(set(pred_files) & set(gt_files))

    all_pixel_acc = []
    all_iou = []
    all_dice = []
    all_prf = []

    print(f"=== Image-wise Evaluation ===\n")

    for file in matching:
        gt_path = os.path.join(gt_dir, file)
        pred_path = os.path.join(pred_dir, file)

        gt_img = np.array(Image.open(gt_path).convert("RGB"))
        pred_img = np.array(Image.open(pred_path).convert("RGB"))

        gt_mask = utils.convert_color_mask_to_class_ids(gt_img, color_to_class)
        pred_mask = utils.convert_color_mask_to_class_ids(pred_img, color_to_class)

        iou, dice, pixel_acc, prf = compute_metrics(gt_mask, pred_mask, num_classes, class_mapping)

        all_pixel_acc.append(pixel_acc)
        all_iou.append(iou)
        all_dice.append(dice)
        all_prf.append(prf)

        print(f"Image: {file}")
        print(f"  Pixel Accuracy: {pixel_acc:.4f}")
        for class_name in class_mapping.values():
            print(f"  Class: {class_name}")
            print(f"    IoU: {iou[class_name]:.4f}")
            print(f"    Dice Score: {dice[class_name]:.4f}")
            print(f"    Precision: {prf[class_name]['Precision']:.4f}")
            print(f"    Recall: {prf[class_name]['Recall']:.4f}")
            print(f"    F1 Score: {prf[class_name]['F1']:.4f}")
        print("")

    # Compute overall mean metrics
    mean_pixel_acc = np.mean(all_pixel_acc)
    mean_iou = {class_name: np.mean([img_iou[class_name] for img_iou in all_iou]) for class_name in class_mapping.values()}
    mean_dice = {class_name: np.mean([img_dice[class_name] for img_dice in all_dice]) for class_name in class_mapping.values()}
    mean_prf = {
        class_name: {
            "Precision": np.mean([img_prf[class_name]["Precision"] for img_prf in all_prf]),
            "Recall": np.mean([img_prf[class_name]["Recall"] for img_prf in all_prf]),
            "F1": np.mean([img_prf[class_name]["F1"] for img_prf in all_prf])
        } for class_name in class_mapping.values()
    }

    print(f"\n=== Overall Evaluation Summary ===\n")
    print(f"Mean Pixel Accuracy over all images: {mean_pixel_acc:.4f}\n")

    for class_name in class_mapping.values():
        print(f"Class: {class_name}")
        print(f"  Mean IoU: {mean_iou[class_name]:.4f}")
        print(f"  Mean Dice Score: {mean_dice[class_name]:.4f}")
        print(f"  Mean Precision: {mean_prf[class_name]['Precision']:.4f}")
        print(f"  Mean Recall: {mean_prf[class_name]['Recall']:.4f}")
        print(f"  Mean F1 Score: {mean_prf[class_name]['F1']:.4f}")
        print("-" * 50)
