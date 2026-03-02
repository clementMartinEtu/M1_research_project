import torch
import matplotlib.pyplot as plt
import numpy as np

import random

import os
from PIL import Image

import sam2
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

print(sam2.__file__)

# load model
checkpoint = "/mnt/c/Users/maner/git/projet-de-recherche-etude-de-biodiversite/segment-anything-2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam = build_sam2(model_cfg, checkpoint, device="cuda")

mask_generator = SAM2AutomaticMaskGenerator(sam)

# load image with Pillow
image_folder_path = "/mnt/d/datasets/Leroy_Merlin/Hypericum_perforatum"
output_mask_folder = "/mnt/c/Users/maner/git/projet-de-recherche-etude-de-biodiversite/output_masks/masks/"
overlay_output_folder = "/mnt/c/Users/maner/git/projet-de-recherche-etude-de-biodiversite/output_masks/mask_overlays/"

os.makedirs(output_mask_folder, exist_ok=True)
os.makedirs(overlay_output_folder, exist_ok=True)

# --- Process each image ---
for filename in os.listdir(image_folder_path):
    if filename.lower().endswith((".jpg", ".png")):
        image_path = os.path.join(image_folder_path, filename)
        image = np.array(Image.open(image_path).convert("RGB"))
        print(f"Processing {filename}, image shape: {image.shape}")

        # Generate SAM2 masks
        masks = mask_generator.generate(image)
        print(f"{filename}: {len(masks)} masks")

        # Save masks as .npy
        mask_path = os.path.join(output_mask_folder, f"{os.path.splitext(filename)[0]}_masks.npy")
        np.save(mask_path, masks)

        # --- Visualization ---
        plt.figure(figsize=(8,8))
        plt.imshow(image)

        for mask in masks:
            m = mask["segmentation"]
            color = (random.random(), random.random(), random.random())
            plt.contour(m, colors=[color], linewidths=0.5)

        plt.axis("off")
        overlay_path = os.path.join(overlay_output_folder, f"{os.path.splitext(filename)[0]}_overlay.png")
        plt.savefig(overlay_path, dpi=300, bbox_inches='tight')
        plt.close()  # close the figure to save memory

        print(f"Saved overlay to {overlay_path}")