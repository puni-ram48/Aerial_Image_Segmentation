import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import numpy as np
import utils

CITYSCAPES_COLORMAP = {
    0: [128, 64, 128], 1: [244, 35, 232], 2: [70, 70, 70], 3: [102, 102, 156], 4: [190, 153, 153],
    5: [153, 153, 153], 6: [250, 170, 30], 7: [220, 220, 0], 8: [107, 142, 35], 9: [152, 251, 152],
    10: [70, 130, 180], 11: [220, 20, 60], 12: [255, 0, 0], 13: [0, 0, 142], 14: [0, 0, 70],
    15: [0, 60, 100], 16: [0, 80, 100], 17: [0, 0, 230], 18: [119, 11, 32]
}

def load_model(device):
    print("Loading SegFormer model...")
    feature_extractor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024").eval()
    model.to(device)
    print(f"Model loaded successfully on {device}")
    return model, feature_extractor

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor()
    ])
    resized = transform(image).permute(1, 2, 0).numpy() * 255
    return resized.astype(np.uint8)

def classify_and_colorize(mask):
    mask[mask == 1] = 0
    mask[mask == 5] = 0
    mask[mask == 7] = 0
    mask[mask == 9] = 8
    mask[mask == 3] = 2
    mask[mask == 4] = 2

    output = np.zeros((1024, 1024, 3), dtype=np.uint8)
    for class_id, color in CITYSCAPES_COLORMAP.items():
        output[mask == class_id] = color

    return output

def generate_masks(images, model, feature_extractor, device):
    inputs = feature_extractor(images=images, return_tensors="pt").to(device)
    with torch.no_grad(), torch.autocast(device_type=device):
        outputs = model(**inputs)
    logits = torch.nn.functional.interpolate(outputs.logits, size=(1024, 1024), mode="bilinear", align_corners=False)
    masks = torch.argmax(logits, dim=1).cpu().numpy()
    return masks

def process_images(input_dir, output_dir, batch_size, device):
    model, feature_extractor = load_model(device)
    utils.ensure_dir(output_dir)
    print(f"Output directory is ready at {output_dir}")

    images = []
    filenames = []

    file_list = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))]
    total_files = len(file_list)
    print(f"Found {total_files} images to process.")

    for idx, file in enumerate(file_list):
        img_path = os.path.join(input_dir, file)
        img = Image.open(img_path).convert("RGB")
        resized = preprocess_image(img)
        images.append(img)
        filenames.append(file)

        if len(images) == batch_size:
            print(f"Processing batch {idx // batch_size + 1}...")
            masks = generate_masks(images, model, feature_extractor, device)
            for i, mask in enumerate(masks):
                final_img = classify_and_colorize(mask)
                out_path = os.path.join(output_dir, filenames[i])
                Image.fromarray(final_img).save(out_path)
                print(f"Saved segmented image: {filenames[i]}")
            images.clear()
            filenames.clear()

    if images:
        print(f"Processing final batch...")
        masks = generate_masks(images, model, feature_extractor, device)
        for i, mask in enumerate(masks):
            final_img = classify_and_colorize(mask)
            out_path = os.path.join(output_dir, filenames[i])
            Image.fromarray(final_img).save(out_path)
            print(f"Saved segmented image: {filenames[i]}")

    print("Segmentation complete! All images saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SegFormer Segmentation Pipeline")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    args = parser.parse_args()

    config = utils.load_config(args.config)

    input_dir = config.get("input_dir")
    output_dir = config.get("output_dir")
    batch_size = config.get("batch_size", 4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    process_images(input_dir, output_dir, batch_size, device)
