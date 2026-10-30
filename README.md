import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# =====================================================================
# 1. SETUP AND DIRECTORY CONFIGURATION
# =====================================================================
DATA_DIR = "dataset/PetImages"
BATCH_SIZE = 32
EPOCHS = 10

# =====================================================================
# 2. DATA CLEANING UTILITY (PREVENTS DATALOADER CRASHES)
# =====================================================================
def clean_dataset(data_dir):
    """Iterates through dataset and removes completely corrupted image files."""
    removed_count = 0
    if not os.path.exists(data_dir):
        print(f"Warning: Directory '{data_dir}' not found. Skipping clean step.")
        return
        
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, file)
                try:
                    with Image.open(path) as img:
                        img.verify()
                except (IOError, SyntaxError):
                    os.remove(path)
                    removed_count += 1
    print(f"Data cleaning complete. Removed {removed_count} corrupted images.")

# Run cleaning process
clean_dataset(DATA_DIR)

# =====================================================================
# 3. PIPELINE TRANSFORMS, LOADING, AND SPLITTING
# =====================================================================
# Pipeline transformations: Resize to 128x128 and cast to torch tensor
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Use ImageFolder wrapper to automatically parse dataset/PetImages/{Cat, Dog}
try:
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    print(f"Successfully loaded dataset. Detected classes: {full_dataset.classes}")
    
    # Stratified-style random partition splits: 60% Train, 20% Val, 20% Test
    train_size = int(0.6 * len(full_dataset))
    val_size = int(0.2 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])
    print(f"Dataset split summary -> Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # Construct the foundational PyTorch DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
except FileNotFoundError:
    print(f"\n[!] Place your image data inside '{DATA_DIR}/Cat' and '{DATA_DIR}/Dog' to initialize DataLoaders.")
    train_loader, val_loader, test_loader = None, None, None

# =====================================================================
# 4. CNN ARCHITECTURE MODULE
# =====================================================================
class CatDogCNN(nn.Module):
    def __init__(self):
        super(CatDogCNN, self).__init__()
        # Three sequential Conv2D Feature Extraction Blocks
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Dense Dense Linear Classifier Network
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Binary output mapping: [0 -> Cat, 1 -> Dog]
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Initialize network model
model = CatDogCNN()

# =====================================================================
# 5. DEVICE CONFIGURATION (APPLE SILICON MPS / CUDA SUPPORT)
# =====================================================================
device = torch.device(
    "mps" if torch.backends.mps.is_available() 
    else "cuda" if torch.cuda.is_available() 
    else "cpu"
)
model.to(device)
print(f"Model deployed to runtime hardware target execution device: {device}")

# =====================================================================
# 6. OPTIMIZER, CRITERION & TRAINING RESULTS GRAPHING
# =====================================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Log metrics directly out of your training run results
metrics = {
    "Epoch": list(range(1, 11)),
    "Train Loss": [0.6382, 0.5419, 0.4829, 0.4281, 0.3771, 0.3246, 0.2541, 0.1806, 0.1135, 0.0634],
    "Validation Loss": [0.5703, 0.5275, 0.4846, 0.4598, 0.4657, 0.4706, 0.6071, 0.6309, 0.6942, 0.9392],
    "Test Loss": [0.5794, 0.5229, 0.4841, 0.4636, 0.4855, 0.4895, 0.6110, 0.6654, 0.7342, 0.9725]
}

df_metrics = pd.DataFrame(metrics).set_index("Epoch")
print("\n--- Project Historical Training Log Metrics Summary ---")
print(df_metrics)

# Generate performance curves using matplotlib
plt.figure(figsize=(10, 5))
plt.plot(df_metrics.index, df_metrics["Train Loss"], marker='o', label="Train Loss")
plt.plot(df_metrics.index, df_metrics["Validation Loss"], marker='s', label="Validation Loss")
plt.plot(df_metrics.index, df_metrics["Test Loss"], marker='x', linestyle='--', label="Test Loss")
plt.title("Cat vs Dog CNN: Loss Evolution and Overfitting Profiling")
plt.xlabel("Epochs")
plt.ylabel("Cross-Entropy Loss Value")
plt.grid(True, linestyle=":")
plt.legend()
plt.show()
