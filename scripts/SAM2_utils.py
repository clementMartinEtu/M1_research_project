import gc

import torch
import numpy as np
import os
import random
import matplotlib.pyplot as plt

from PIL import Image

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def load_sam2_model(checkpoint, model_cfg, device="cuda"):
    """
    Load SAM2 model and return mask generator
    """
    sam = build_sam2(model_cfg, checkpoint, device=device)

    mask_generator = SAM2AutomaticMaskGenerator(
        sam,
        points_per_side=16,
        pred_iou_thresh=0.9,
        stability_score_thresh=0.9
    )

    return mask_generator


def process_image(image_path, mask_generator):
    """
    Generate masks for one image and return best mask
    """
    pil_image = Image.open(image_path).convert("RGB")

    # convert to numpy
    image = np.array(pil_image)

    with torch.no_grad():
        masks = mask_generator.generate(image)

    if len(masks) == 0:
        return image, None, []

    best_mask = max(masks, key=lambda m: m["predicted_iou"])

    return image, best_mask, masks


def save_overlay(image, masks, output_path):
    """
    Save overlay visualization
    """
    plt.figure(figsize=(8,8))
    plt.imshow(image)

    for mask in masks:
        m = mask["segmentation"]
        color = (random.random(), random.random(), random.random())
        plt.contour(m, colors=[color], linewidths=0.5)

    plt.axis("off")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def process_folder(image_folder, mask_generator, mask_output, overlay_output, only_best_mask=False):

    os.makedirs(mask_output, exist_ok=True)
    os.makedirs(overlay_output, exist_ok=True)

    for filename in os.listdir(image_folder):

        if not filename.lower().endswith((".jpg", ".png")):
            continue

        image_path = os.path.join(image_folder, filename)

        print(f"Processing {filename}")

        image, best_mask, masks = process_image(image_path, mask_generator)

        if best_mask is None:
            print("No mask detected")
            continue


        # save masks
        if only_best_mask:
            masks = [best_mask]

        for m in masks:
            m["best_masks"] = []
   
        mask_path = os.path.join(
            mask_output,
            f"{os.path.splitext(filename)[0]}_masks.npy"
        )

        np.save(mask_path, masks)

        # save overlay
        overlay_path = os.path.join(
            overlay_output,
            f"{os.path.splitext(filename)[0]}_overlay.png"
        )

        save_overlay(image, masks, overlay_path)

        print(f"Saved overlay → {overlay_path}")

        #save memory
        torch.cuda.empty_cache()
        gc.collect()