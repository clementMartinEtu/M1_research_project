import numpy as np
from PIL import Image
import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm


def extract_mask_crop(image, mask):
    """
    Extract a bounding-box crop from an image using a SAM2 mask dictionary.

    Args:
        image (np.ndarray): Source RGB image of shape (H, W, 3).
        mask (dict): SAM2 mask dict containing a 'bbox' key with [x, y, w, h].

    Returns:
        np.ndarray: Cropped region of the image corresponding to the mask bounding box.
    """
    x, y, w, h = mask["bbox"]
    x, y, w, h = int(x), int(y), int(w), int(h)
    crop = image[y:y+h, x:x+w]
    return crop


def extract_all_crops(image, masks):
    """
    Extract bounding-box crops for all masks in a list.

    Args:
        image (np.ndarray): Source RGB image of shape (H, W, 3).
        masks (list[dict]): List of SAM2 mask dicts.

    Returns:
        list[np.ndarray]: One crop per mask.
    """
    crops = []
    for m in masks:
        crops.append(extract_mask_crop(image, m))
    return crops


def save_crops(crops, output_folder, image_name="crop"):
    """
    Save a list of image crops as numbered PNG files to an output folder.

    Args:
        crops (list[np.ndarray]): List of image crops as NumPy arrays.
        output_folder (str): Path to the output directory.
        image_name (str): Base name for saved files. Files are named {image_name}_{i}.png.
    """
    os.makedirs(output_folder, exist_ok=True)
    for i, crop in enumerate(crops):
        image = Image.fromarray(crop)
        image.save(os.path.join(output_folder, f"{image_name}_{i}.png"))
    print(f"{len(crops)} crops saved to {output_folder}")


def process_mask_files(image_dir, mask_dir, crop_output):
    """
    Process all .npy mask files in a directory and extract crops for each image.

    Args:
        image_dir (str): Directory containing the original .jpg images.
        mask_dir (str): Directory containing the SAM2 .npy mask files.
        crop_output (str): Root directory where crop subfolders will be created.
    """
    os.makedirs(crop_output, exist_ok=True)

    for mask_file in os.listdir(mask_dir):
        if not mask_file.endswith(".npy"):
            continue

        mask_path = os.path.join(mask_dir, mask_file)
        masks = np.load(mask_path, allow_pickle=True)

        image_name = mask_file.replace("_masks.npy", ".jpg")
        image_path = os.path.join(image_dir, image_name)
        print(image_path)

        if not os.path.exists(image_path):
            print(f"Image not found for {mask_file}")
            continue

        image = np.array(Image.open(image_path).convert("RGB"))
        flowers_id = image_name.split(".")[0]
        crops = extract_all_crops(image, masks)
        save_crops(crops, os.path.join(crop_output, flowers_id), flowers_id)


def retrieve_right_masks(PATH_FLOWERS="../data/flowers/", PATH_MASKS="../output_masks/"):
    """
    Filter original SAM2 .npy mask arrays using CNN-accepted crop filenames.

    Args:
        PATH_FLOWERS (str): Directory containing accepted crop PNGs.
        PATH_MASKS (str): Root directory containing masks/ and masks_filtered/ subdirectories.
    """
    crops_by_id = {}
    for crop in os.listdir(PATH_FLOWERS):
        if not crop.endswith(".png"):
            continue

        name, flower_id, number = os.path.splitext(crop)[0].split("_")
        key = f"{name}_{flower_id}"
        if key not in crops_by_id:
            crops_by_id[key] = []
        crops_by_id[key].append(int(number))

    for name_id, numbers in crops_by_id.items():
        mask_file = os.path.join(PATH_MASKS, "masks", f"{name_id}_masks.npy")
        masks = np.load(mask_file, allow_pickle=True)

        # stock indices that must be kept
        indices = np.zeros(len(masks), dtype=bool)
        indices[numbers] = True
        masks_kept = masks[indices]

        PATH_MASKS_FILTERED = os.path.join(PATH_MASKS, "masks_filtered")
        os.makedirs(PATH_MASKS_FILTERED, exist_ok=True)

        mask_filtered_path = os.path.join(PATH_MASKS_FILTERED, f"{name_id}_masks.npy")

        # save filtered masks
        np.save(mask_filtered_path, masks_kept)


def create_dataset(batch_size=32, seed=42):
    """
    Create train and validation DataLoaders from the CNN dataset.

    Args:
        batch_size (int): Number of samples per batch.
        seed (int): Random seed for reproducible splits.

    Returns:
        tuple[DataLoader, DataLoader]: (train_loader, test_loader) with an 80/20 split.
    """
    transform = transforms.Compose([
        transforms.Pad(25),
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = datasets.ImageFolder("../data/CNN_dataset", transform=transform)
    torch.manual_seed(seed=seed)

    # split into train and test
    # the train dataset have 80% of the data and the test dataset have 20% of the data
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train(train_loader, test_loader, optimizer, criterion, model, epochs= 10, device="cpu"):
    """
    Train a binary classification model with validation, LR scheduling,
    and automatic best-model checkpointing.

    Args:
        train_loader (DataLoader): DataLoader for the training set.
        test_loader (DataLoader): DataLoader for the validation set.
        optimizer (torch.optim.Optimizer): PyTorch optimizer.
        criterion (nn.Module): Loss function — must be BCEWithLogitsLoss.
        model (nn.Module): PyTorch model instance.
        epochs (int): Number of training epochs.
        device (str): 'cuda' or 'cpu'.

    Returns:
        dict: Dictionary with 'train_loss', 'val_loss', and 'val_acc' lists.
    """
    model.to(device)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    best_val_loss = float('inf')

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    epoch_bar = tqdm(range(epochs), desc="Training", unit="epoch")

    for epoch in epoch_bar:

        # Training phase
        model.train()
        train_loss = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False, unit="batch")
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        val_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  ", leave=False, unit="batch")
        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                labels = labels.float().unsqueeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                predicted = (outputs >= 0.0).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                val_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        avg_val_acc = 100 * correct / total

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(avg_val_acc)

        # Update outer epoch bar with final metrics
        epoch_bar.set_postfix(
            train_loss=f"{avg_train_loss:.4f}",
            val_loss=f"{avg_val_loss:.4f}",
            val_acc=f"{avg_val_acc:.1f}%",
            best=f"{best_val_loss:.4f}"
        )

    return history


if __name__ == "__main__":
    from sys import argv

    if len(argv) == 2:
        IMAGE_DIR = argv[1]
    else:
        print("Usage: python mask_classifier.py <image_directory>")
        exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    PATH_DIR = "../output_masks"
    MASK_DIR = os.path.join(PATH_DIR, "masks")
    CROP_OUTPUT = os.path.join(PATH_DIR, "mask_crops")

    process_mask_files(IMAGE_DIR, MASK_DIR, CROP_OUTPUT)