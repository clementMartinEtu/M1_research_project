# Research Project:

# Flower Mask Filtering Pipeline

Automated pipeline for extracting and filtering flower segmentation masks from biodiversity images, combining SAM2 zero-shot segmentation with a lightweight CNN classifier.

---

## How It Works

The pipeline operates in four sequential stages:

**1. Segmentation — `SAM2_utils.py`**  
Each input image is processed by SAM2 in automatic mask generation mode, producing a set of candidate segmentation masks covering all detected regions in the image.

**2. Crop Extraction — `mask_classifier.py`**  
For each mask, a bounding-box crop is extracted from the original image and saved as a PNG. This produces a flat collection of small image patches, one per candidate mask.

**3. Classification — SmallCNN**  
A custom binary CNN classifier (`SmallCNN`) is trained on labeled flower / non-flower crops. Once trained, it runs inference on all extracted crops and keeps only those predicted as flowers.

The dataset for training the CNN is organized as follows:
```CNN_dataset/
├── flowers/
│   ├── ... (flower crops)
├── not_flowers/
│   ├── ... (non-flower crops)
```

It contains 450 Hypericum perforatum crops and 550 non-flower crops, split into 80% training and 20% validation sets.

**4. Mask Filtering — `mask_classifier.py`**  
The accepted crop indices are used to filter the original SAM2 `.npy` mask arrays, producing a clean set of flower-only masks saved to `output_masks/masks_filtered/`.
```
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
│   ├── SAM2_utils.py         # SAM2 loading and overlay utilities
│   └── mask_classifier.py    # Crop extraction, CNN training, mask filtering
├── notebooks/
│   └── sam2_test.ipynb       # Full pipeline notebook
├── requirements.txt
└── README.md
```

---

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/your-username/projet-de-recherche-etude-de-biodiversite.git
cd projet-de-recherche-etude-de-biodiversite
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
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

## Training

Training is handled in `notebooks/sam2_test.ipynb`. Run the notebook cells in order:

**Step 1 — Segment images with SAM2**  
Loads the SAM2 model and runs automatic mask generation on all images in the input folder. Saves raw `.npy` mask files and overlay visualizations.

**Step 2 — Extract crops**  
Calls `process_mask_files()` to extract a bounding-box crop for each SAM2 mask and save them as PNG files organized by image.

**Step 3 — Train the CNN classifier**  
Builds the `CNN_dataset` from labeled flower / non-flower crops, then trains `SmallCNN` using:
- Loss: `BCEWithLogitsLoss`
- Optimizer: `Adam` (lr=0.001, weight_decay=1e-4)
- Scheduler: `ReduceLROnPlateau` (patience=3, factor=0.5)
- Best model automatically saved to `best_model.pth`

**Step 4 — Filter masks**  
Loads `best_model.pth`, runs inference on all crops, copies accepted crops to the flowers directory, then calls `retrieve_right_masks()` to produce filtered `.npy` mask files.

---

## Results

The final cell of the notebook displays a side-by-side comparison for each test image:

- **Left** — original image with all SAM2 masks overlaid
- **Right** — same image with only CNN-approved flower masks

Below is an example result on *Hypericum perforatum*:

<!-- Replace with your actual result image -->
![Result](output_masks/test/filtered_mask_overlays/Hypericum_perforatum_example_overlay.png)

The CNN reached **95.8% validation accuracy** after 30 epochs on the flower / non-flower classification task, significantly reducing the number of spurious masks in the final output.