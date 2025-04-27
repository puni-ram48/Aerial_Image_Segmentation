import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import utils

def visualize_masks(matching_files, pred_dir, gt_dir, num_samples=5):
    print(f"Visualizing {num_samples} samples...")
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, num_samples * 5))

    for i, file in enumerate(matching_files[:num_samples]):
        pred = np.array(Image.open(os.path.join(pred_dir, file)).convert("RGB"))
        gt = np.array(Image.open(os.path.join(gt_dir, file)).convert("RGB"))

        axes[i, 0].imshow(pred)
        axes[i, 0].set_title(f"Predicted - {file}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt)
        axes[i, 1].set_title(f"Ground Truth - {file}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Segmentation Masks")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    args = parser.parse_args()

    config = utils.load_config(args.config)
    pred_dir = config.get("output_dir")
    gt_dir = config.get("gt_dir")
    num_samples = config.get("num_samples", 5)

    pred_files = sorted(os.listdir(pred_dir))
    gt_files = sorted(os.listdir(gt_dir))
    matching = list(set(pred_files) & set(gt_files))

    visualize_masks(matching, pred_dir, gt_dir, num_samples)
