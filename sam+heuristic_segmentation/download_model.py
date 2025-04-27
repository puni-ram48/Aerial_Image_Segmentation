"""
Script to download the SAM model checkpoint if not available locally.
"""

import os
import requests

def download_sam_checkpoint(url, save_path):
    """
    Download a model checkpoint from a URL.

    Args:
        url (str): URL to download from.
        save_path (str): Where to save the checkpoint.
    """
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f" Model checkpoint saved to {save_path}")
    else:
        print("Failed to download checkpoint")

if __name__ == "__main__":
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
    save_path = "./saved_models/sam_vit_l.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    download_sam_checkpoint(url, save_path)
