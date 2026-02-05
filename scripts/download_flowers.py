import pandas as pd
import requests
import os
from PIL import Image
from io import BytesIO
# from concurrent.futures import ThreadPoolExecutor, as_completed

def is_right_size(image):
    """
    Check if the image has the correct dimensions (1024x768 or 768x1024).
    
    :param image: PIL Image object
    """
    width, height = image.size
    return (width == 1024 and height == 768) or (width == 768 and height == 1024)

def download_image(row, IMAGE_DIR):
    """
    Download an image based on information provided in a data row.

    The function creates a directory for each species if it does not exist,
    then downloads the corresponding image in large format and saves it locally.

    Parameters
    ----------
    row : dict or pandas.Series
        Data row containing at least:
        - "species" : species name
        - "image_id" : image identifier
        - "url" : image URL (medium will be replaced by large)

    IMAGE_DIR : str
        Root directory where images will be stored.

    Returns
    -------
    str or None
        "already downloaded" if the image already exists,
        "error: <message>" if an error occurs,
        None if the download succeeds.
    """
    species_dir = os.path.join(IMAGE_DIR, row["species"].replace(" ", "_"))
    os.makedirs(species_dir, exist_ok=True)
    print (f'Downloading image {row["image_id"]} of species {row["species"]}')

    filename = f'{row["species"]}_{row["image_id"]}.jpg'
    filepath = os.path.join(species_dir, filename)

    if os.path.exists(filepath):
        return "already downloaded"

    try:
        url = row["url"].replace("medium", "large")
        r = requests.get(url, stream=True)
        r.raise_for_status()

        img = Image.open(BytesIO(r.content))
        if is_right_size(img):
            with open(filepath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            return f"image skipped: image size {img.size} is not 1024x768 or 768x1024"
            
    except Exception as e:
        return f"error: {e}"

if __name__ == "__main__":
    from sys import argv

    if len(argv) == 2:
        PATH_DIR = argv[0]
        IMAGE_DIR = argv[1]
    else:
        print("Usage: python download_flowers.py <path_to_csv> <image_directory>")
        exit(1)

    os.makedirs(IMAGE_DIR, exist_ok=True)

    df = pd.read_csv(
    PATH_DIR,
    header=None,
    names=["image_id","user_name", "license", "url", "species"]
    )

    #download images
    for i in range(1, min(len(df), 50)):  # change range as needed

        result = download_image(df.iloc[i], IMAGE_DIR)
        print(f'Image {i}/{len(df)-1}: Done')