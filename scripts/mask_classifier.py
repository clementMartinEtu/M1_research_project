import numpy as np
from PIL import Image
import os
import torch
from torchvision import transforms, datasets
from torchvision.models import models
from torch.utils.data import DataLoader


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

def retrieve_right_masks():

    PATH_FLOWERS = "../data/flowers/"

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

def create_dataset():

    dataset = datasets.ImageFolder(
        "../data/CNN_dataset",
        transform=transforms.ToTensor()
    )

    # split into train and test
    # the train dataset have 80% of the data and the test dataset have 20% of the data
    traindata, testdata = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset) - int(0.8*len(dataset))])
    train_loader = DataLoader(traindata, batch_size=32, shuffle=True)
    test_loader = DataLoader(testdata, batch_size=32, shuffle=False)

    return train_loader, test_loader

def train_model(model):

    # model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer for binary classification
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(512, 1),
        torch.nn.Sigmoid()
    )

    return model

def train(train_loader, test_loader, optimizer, criterion, model, epochs=10, device="cpu"):

    model.to(device)

    for epoch in range(epochs):

        # --- Training ---
        model.train()
        train_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # --- Validation ---
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                labels = labels.float().unsqueeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                predicted = (outputs >= 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        print(f"Epoch {epoch+1}/{epochs} "
              f"| Train Loss: {train_loss/len(train_loader):.4f} "
              f"| Val Loss: {val_loss/len(test_loader):.4f} "
              f"| Val Acc: {100*correct/total:.1f}%")



        

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







    