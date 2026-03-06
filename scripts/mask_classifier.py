import numpy as np
from PIL import Image
import os

def extract_mask_crop(image, mask):
    
    m = mask["segmentation"]

    ys, xs = np.where(m)

    if len(xs) == 0 or len(ys) == 0:
        return None

    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    crop = image[y1:y2, x1:x2]

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



    