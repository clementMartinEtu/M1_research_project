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
    Load a SAM2 model and initialize an automatic mask generator.

    Args:
        checkpoint (str): Path to the model checkpoint file.
        model_cfg (str or dict): SAM2 model configuration.
        device (str, optional): Device to run the model on ("cuda" or "cpu").
                                Defaults to "cuda".

    Returns:
        SAM2AutomaticMaskGenerator: A configured mask generator.

    Notes:
        - High thresholds (0.9) favor high-quality masks.
        - Lower thresholds may increase the number of detected masks
          but reduce precision.
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
    Generate segmentation masks for a single image.

    Args:
        image_path (str): Path to the input image.
        mask_generator (SAM2AutomaticMaskGenerator): Mask generator instance.

    Returns:
        tuple:
            - image (np.ndarray): Image as a NumPy array (H, W, 3).
            - best_mask (dict or None): Mask with the highest predicted IoU.
            - masks (list): List of all generated masks.

    Notes:
        - Returns (image, None, []) if no masks are detected.
        - The best mask is selected using the "predicted_iou" score.
    """
    pil_image = Image.open(image_path).convert("RGB")

    # Convert to NumPy array
    image = np.array(pil_image)

    with torch.no_grad():
        masks = mask_generator.generate(image)

    if len(masks) == 0:
        return image, None, []

    best_mask = max(masks, key=lambda m: m["predicted_iou"])

    return image, best_mask, masks


def save_overlay(image, masks, output_path, linewidth=0.5):
    """
    Save an image with segmentation mask contours overlaid.

    Args:
        image (np.ndarray): Input image (H, W, 3).
        masks (list): List of mask dictionaries.
        output_path (str): Path to save the overlay image.
        linewidth (float, optional): Width of contour lines. Defaults to 0.5.

    Notes:
        - Each mask is displayed with a random color.
        - Output resolution is in average 1600x1600 pixels.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image)

    for mask in masks:
        m = mask["segmentation"]
        color = (random.random(), random.random(), random.random())
        ax.contour(m, colors=[color], linewidths=linewidth)

    ax.axis("off")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")

    # Free matplotlib memory
    fig.clf()
    plt.close(fig)
    plt.close("all")


def process_folder(
    image_folder,
    mask_generator,
    mask_output,
    overlay_output,
    only_best_mask=False,
    linewidth=0.5
):
    """
    Process all images in a folder and generate segmentation masks.

    Args:
        image_folder (str): Directory containing input images.
        mask_generator (SAM2AutomaticMaskGenerator): Mask generator instance.
        mask_output (str): Directory to save mask files (.npy).
        overlay_output (str): Directory to save overlay images (.png).
        only_best_mask (bool, optional): If True, save only the best mask.
                                         Defaults to False.
        linewidth (float, optional): Contour thickness for overlays.

    Workflow:
        1. Iterate over all .jpg and .png images.
        2. Generate masks using SAM2.
        3. Save masks as NumPy files.
        4. Save overlay visualizations.

    Notes:
        - Skips non-image files.
        - Skips images with no detected masks.
        - Frees GPU and CPU memory after each image.
    """
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

        # Optionally keep only the best mask
        if only_best_mask:
            masks = [best_mask]

        mask_path = os.path.join(
            mask_output,
            f"{os.path.splitext(filename)[0]}_masks.npy"
        )
        np.save(mask_path, masks)

        overlay_path = os.path.join(
            overlay_output,
            f"{os.path.splitext(filename)[0]}_overlay.png"
        )

        save_overlay(image, masks, overlay_path, linewidth=linewidth)
        plt.close("all")

        print(f"Saved overlay → {overlay_path}")

        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()