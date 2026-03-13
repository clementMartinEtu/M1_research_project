from detectron2.projects.DensePose.densepose.modeling.losses import mask
import numpy as np
from PIL import Image
import os
import torch


def extract_mask_crop(image, mask):

    x, y, w, h = mask["bbox"]

    x, y, w, h = int(x), int(y), int(w), int(h)

    crop = image[y:y+h, x:x+w]

    return crop

def extract_all_crops(image, masks):

    crops = []
    for m in masks:
        crops.append(extract_mask_crop(image, m))

    return crops

def save_crops(crops, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    for i, crop in enumerate(crops):
        image = Image.fromarray(crop)
        image.save(os.path.join(output_folder, f"mask_{i}.png"))

    print(f"{len(crops)} crops saved to {output_folder}")

def process_mask_files(image_dir, mask_dir, crop_output):

    os.makedirs(crop_output, exist_ok=True)

    for mask_file in os.listdir(mask_dir):

        if not mask_file.endswith(".npy"):
            continue

        mask_path = os.path.join(mask_dir, mask_file)

        # load masks
        masks = np.load(mask_path, allow_pickle=True)

        # recover image name
        image_name = mask_file.replace("_masks.npy", ".jpg")

        image_path = os.path.join(image_dir, image_name)
        print(image_path)

        if not os.path.exists(image_path):
            print(f"Image not found for {mask_file}")
            continue

        image = np.array(Image.open(image_path).convert("RGB"))

        crops = extract_all_crops(image, masks)

        save_crops(crops, os.path.join(crop_output, image_name.split(".")[0]))

def retrieve_right_masks():

    PATH_FLOWERS = "../data/flowers/"

    crops_by_id = {}
    for crop in os.listdir(PATH_FLOWERS):

        if not crop.endswith(".png"):
            continue
        
        name, id, number = os.path.splitext(crop)[0].split("_")

        crops_by_id[f"{name}_{id}"].append(int(number))

    for name_id, numbers in crops_by_id.items():

        mask_file = os.path.join("../output_masks/masks", f"{name_id}.npy")

        # load masks
        masks = np.load(mask_file, allow_pickle=True)

        # stock indices that must be kept
        indices = np.zeros(len(masks), dtype=bool)
        indices[numbers] = True

        masks_kept = masks[indices]

        os.makedirs("../output_masks/masks_filtered", exist_ok=True)

        mask_filtered_path = os.path.join("../output_masks/masks_filtered", f"{name_id}.npy")

        # save filtered masks
        np.save(mask_filtered_path, masks_kept)
        

if __name__ == "__main__":
    from sys import argv

    if len(argv) == 4:
        PATH_DIR = argv[1]
        IMAGE_DIR = argv[2]
        MASK_OUTPUT_DIR = argv[3]
    else:
        print("Usage: python mask_classifier.py <path_to_csv> <image_directory> <mask_output_directory>")
        exit(1)
    

    # Use the correct config path format
    sam2_checkpoint = "/mnt/c/Users/maner/git/projet-de-recherche-etude-de-biodiversite/segment-anything-2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    MASK_DIR = os.path.join(PATH_DIR, "masks")
    CROP_OUTPUT = os.path.join(PATH_DIR, "mask_crops")

    process_mask_files(
        IMAGE_DIR,
        MASK_DIR,
        CROP_OUTPUT
    )







    