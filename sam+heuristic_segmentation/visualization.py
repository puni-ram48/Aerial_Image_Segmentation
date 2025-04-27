
"""
Visualization script to display ground truth and predicted segmentation side-by-side.
"""

import os
import cv2
import matplotlib.pyplot as plt
from config import config

def display_all_images(num_images=10):
    """
    Display ground truth and predicted images side-by-side.

    Args:
        num_images (int): Number of images to display.
    """
    gt_dir = config["gt_dir"]
    pred_dir = config["output_dir"]

    gt_images = sorted(os.listdir(gt_dir))
    pred_images = sorted(os.listdir(pred_dir))

    num_images = min(num_images, len(gt_images), len(pred_images))

    plt.figure(figsize=(15, num_images * 3))

    for i in range(num_images):
        gt_path = os.path.join(gt_dir, gt_images[i])
        pred_path = os.path.join(pred_dir, pred_images[i])

        gt_img = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB)
        pred_img = cv2.cvtColor(cv2.imread(pred_path), cv2.COLOR_BGR2RGB)

        plt.subplot(num_images, 2, 2 * i + 1)
        plt.imshow(gt_img)
        plt.title(f"Ground Truth: {gt_images[i]}")
        plt.axis("off")

        plt.subplot(num_images, 2, 2 * i + 2)
        plt.imshow(pred_img)
        plt.title(f"Predicted: {pred_images[i]}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    display_all_images(num_images=config["batch_size"])
