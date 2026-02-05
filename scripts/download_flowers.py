import pandas as pd
import requests
import os
# from concurrent.futures import ThreadPoolExecutor, as_completed

def download_image(row, IMAGE_DIR):
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

        with open(filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
            
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