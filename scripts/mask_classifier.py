import numpy as np
from PIL import Image
import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from tqdm import tqdm

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

def save_crops(crops, output_folder, image_name="crop"):

    os.makedirs(output_folder, exist_ok=True)

    for i, crop in enumerate(crops):
        image = Image.fromarray(crop)
        image.save(os.path.join(output_folder, f"{image_name}_{i}.png"))

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

        flowers_id = image_name.split(".")[0]

        crops = extract_all_crops(image, masks)

        save_crops(crops, os.path.join(crop_output, flowers_id), flowers_id)

def retrieve_right_masks(PATH_FLOWERS="../data/flowers/", PATH_MASKS="../output_masks/"):

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
        
        mask_file = os.path.join(PATH_MASKS,"masks", f"{name_id}_masks.npy")

        # load masks
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

    transform = transforms.Compose([
        transforms.Pad(25),
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])

    dataset = datasets.ImageFolder(
        "../data/CNN_dataset",
        transform=transform
    )

    torch.manual_seed(seed=seed)

    # split into train and test
    # the train dataset have 80% of the data and the test dataset have 20% of the data
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train(train_loader, test_loader, optimizer, criterion, model, epochs=10, device="cpu"):

    model.to(device)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    best_val_loss = float('inf')

    epoch_bar = tqdm(range(epochs), desc="Training", unit="epoch")

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in epoch_bar:

        # --- Training ---
        model.train()
        train_loss = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", 
                         leave=False, unit="batch")
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Live loss update on the training bar
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        # --- Validation ---
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        val_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  ", 
                       leave=False, unit="batch")
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

                # Live loss update on the validation bar
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
            val_acc=f"{100*correct/total:.1f}%",
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

    process_mask_files(
        IMAGE_DIR,
        MASK_DIR,
        CROP_OUTPUT
    )







    