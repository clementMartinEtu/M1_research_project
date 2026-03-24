# Research Project:

# Flower Mask Filtering Pipeline

Automated pipeline for extracting and filtering flower segmentation masks from biodiversity images, combining SAM2 zero-shot segmentation with a lightweight CNN classifier.

---

## How It Works

The pipeline operates in four sequential stages:

**1. Data Collection — `download_flowers.py`**  
Downloads flower images from a CSV file containing species metadata and image URLs. Only images with valid dimensions (1024×768 or 768×1024) are kept.

**2. Segmentation — `SAM2_utils.py`**  
Each input image is processed by SAM2 in automatic mask generation mode, producing a set of candidate segmentation masks covering all detected regions in the image.

**3. Crop Extraction — `mask_classifier.py`**  
For each mask, a bounding-box crop is extracted from the original image and saved as a PNG. This produces a flat collection of small image patches, one per candidate mask.

**4. Classification — SmallCNN**  
A custom binary CNN classifier (`SmallCNN`) is trained on labeled flower / non-flower crops. Once trained, it runs inference on all extracted crops and keeps only those predicted as flowers.

The dataset for training the CNN is organized as follows:
```CNN_dataset/
├── flowers/
│   ├── ... (flower crops)
├── not_flowers/
│   ├── ... (non-flower crops)
```

It contains 450 Hypericum perforatum crops and 550 non-flower crops, split into 80% training and 20% validation sets. Those crops was sorted by hand from the initial SAM2 output, which contained many false positives (e.g. leaves, soil patches, etc.).

**4. Mask Filtering — `mask_classifier.py`**  
The accepted crop indices are used to filter the original SAM2 `.npy` mask arrays, producing a clean set of flower-only masks saved to `output_masks/masks_filtered/`.
```
CSV (species metadata + URLs)
     ↓  download_flowers.py
Input images
     ↓  SAM2
All masks (.npy) + All crops (.png)
     ↓  SmallCNN
Accepted crops
     ↓  retrieve_right_masks()
Filtered masks (.npy)
```

---

## Project Structure
```
.
├── data/
│   ├── CNN_dataset/          # Labeled training data (flowers / not_flowers)
│   └── test/                 # Test images
├── output_masks/
│   ├── masks/                # Raw SAM2 .npy mask files
│   ├── mask_overlays/        # Visualization of all SAM2 masks
│   ├── mask_crops/           # Extracted bounding-box crops
│   ├── masks_filtered/       # Filtered .npy masks (flowers only)
│   └── filtered_mask_overlays/  # Visualization of filtered masks
├── scripts/
│   ├── CNN.py                # SmallCNN architecture
│   ├── download_flowers.py   # Image downloading script
│   ├── SAM2_utils.py         # SAM2 loading and overlay utilities
│   └── mask_classifier.py    # Crop extraction, CNN training, mask filtering
├── notebooks/
│   ├── sam2_test.ipynb       # SAM2 segmentation test notebook
│   └── filter_train.ipynb      # Full training notebook
├── requirements.txt
└── README.md
```

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/your-username/projet-de-recherche-etude-de-biodiversite.git
cd projet-de-recherche-etude-de-biodiversite
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download the SAM2 checkpoint**
```bash
mkdir -p checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
     -O checkpoints/sam2.1_hiera_large.pt
```

---

## Data Collection

Images are sourced from a CSV file with the following columns that can be created from [iNaturalist](https://www.inaturalist.org/observations/export?verifiable=true&page=1&spam=false&place_id=any&user_id=&project_id=):

| Column | Description |
|---|---|
| `image_id` | Unique image identifier |
| `user_name` | Contributor username |
| `license` | Image license |
| `url` | Image URL |
| `species` | Flower species name |

Run the downloader with:
```bash
python scripts/download_flowers.py  
```

For example:
```bash
python scripts/download_flowers.py data/Hypericum_perforatum.csv data/images/
```

Images are saved to `<image_directory>/<species_name>/` and only kept if they match the expected dimensions (1024×768 or 768×1024). The script skips already downloaded images and caps downloads at 600 by default (`MAX_DOWNLOADS`).

Those arbitrary sizes as been selected to match with a future work using a Mask R-CNN architecture with a great quality.

If you want to deal with a massive number of masks to crop, you could use this command:

```bash
python mask_classifier.py <image_directory>
```

`<image_directory>` should be the past the folder where the original images are stored, it means that the images without any mask since we use those image as a basis to crop the masks.

---

## Training

Training is handled in `notebooks/filter_train.ipynb`. Run the notebook cells in order:

**Step 1 — Segment images with SAM2**  
Loads the SAM2 model and runs automatic mask generation on all images in the input folder. Saves raw `.npy` mask files and overlay visualizations.

**Step 2 — Extract crops**  
Calls `process_mask_files()` to extract a bounding-box crop for each SAM2 mask and save them as PNG files organized by image.

Each crop as a name like `<image_id>_<mask_id>.png`, where `mask_id` is the index of the mask in the original `.npy` file and `image_id` is the name of the flower image and its id. This allows us to link each crop back to its original image and mask for later filtering.

**Step 3 — Train the CNN classifier**  
Builds the `CNN_dataset` from labeled flower / non-flower crops, then trains `SmallCNN` using:
- Loss: `BCEWithLogitsLoss`
- Optimizer: `Adam` (learning_rate=0.001, weight_decay=1e-4)
- Scheduler: `ReduceLROnPlateau`
- Best model automatically saved to `best_model.pth`

**Step 4 — Filter masks**  
Loads `best_model.pth`, runs inference on all crops, copies accepted crops to the flowers directory, then calls `retrieve_right_masks()` to produce filtered `.npy` mask files.

---

### Large-scale segmentation on Kaggle

Running SAM2 on more than 300 images locally requires significant GPU memory. For this reason, large-scale segmentation was run on Kaggle using `notebooks/sam2_kaggle.ipynb`, which provides free access to a GPU environment with sufficient RAM.

The Kaggle notebook:
- Clones and installs the SAM2 repository directly in the Kaggle environment
- Downloads the `sam2.1_hiera_large` checkpoint
- Reads images from a Kaggle dataset uploaded beforehand
- Runs `process_folder()` on each species folder and saves `.npy` masks and overlay images
- Outputs are then downloaded in the output_mask folder and used as input for the CNN filtering step

To reproduce, upload your images as a Kaggle dataset, import `sam2_kaggle.ipynb`, and update the `image_folder` path to match your dataset location:
```python
image_folder = "/kaggle/input/datasets///"
```

## Results

The final cell of the notebook displays a side-by-side comparison for each test image:

- **Left** — original image with all SAM2 masks overlaid
- **Right** — same image with only CNN-approved flower masks

The CNN reached **95.8% validation accuracy** after 30 epochs on the flower / non-flower classification task, significantly reducing the number of badly classified masks in the final output.