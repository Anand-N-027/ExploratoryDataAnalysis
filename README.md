{
"cells": [
{
"cell_type": "markdown",
"metadata": {},
"source": [
"# Cat vs Dog Image Classification using PyTorch and TorchVision\n",
"\n",
"## Overview\n",
"\n",
"This project implements a Convolutional Neural Network (CNN) from scratch using PyTorch for binary image classification of cats and dogs. The workflow includes image validation, preprocessing, dataset splitting, model development, training, and evaluation.\n",
"\n",
"The project demonstrates fundamental computer vision and deep learning concepts, including:\n",
"\n",
"- Image preprocessing using TorchVision\n",
"- Dataset cleaning and validation\n",
"- Convolutional Neural Networks (CNNs)\n",
"- Train, validation, and test set generation\n",
"- Model training using PyTorch\n",
"- Binary image classification\n",
"- Apple Silicon GPU (MPS) support"
]
},
{
"cell_type": "markdown",
"metadata": {},
"source": [
"---\n",
"\n",
"## Directory Structure\n",
"\n",
"text\n", ".\n", "├── notebook.ipynb\n", "├── README.md\n", "└── dataset/\n", "    └── PetImages/\n", "        ├── Cat/\n", "        │   ├── 0.jpg\n", "        │   ├── 1.jpg\n", "        │   └── ...\n", "        └── Dog/\n", "            ├── 0.jpg\n", "            ├── 1.jpg\n", "            └── ...\n", ""
]
},
{
"cell_type": "markdown",
"metadata": {},
"source": [
"---\n",
"\n",
"## Dataset Description\n",
"\n",
"### dataset/PetImages\n",
"\n",
"Contains image folders for the two target classes:\n",
"\n",
"- Cat\n",
"- Dog\n",
"\n",
"The dataset follows the directory structure required by torchvision.datasets.ImageFolder.\n",
"\n",
"### notebook.ipynb\n",
"\n",
"Contains the complete machine learning pipeline:\n",
"\n",
"- Dataset cleaning\n",
"- Data transformation\n",
"- Data loading\n",
"- CNN architecture\n",
"- Model training\n",
"- Evaluation"
]
},
{
"cell_type": "markdown",
"metadata": {},
"source": [
"---\n",
"\n",
"## Requirements\n",
"\n",
"The project was developed using:\n",
"\n",
"- Python 3.11\n",
"- PyTorch 2.2.2\n",
"- TorchVision 0.17.2\n",
"- NumPy 1.26.4\n",
"- Pandas 2.3.3\n",
"- Matplotlib 3.10.8\n",
"- Pillow 11.3.0\n",
"\n",
"Install dependencies:"
]
},
{
"cell_type": "code",
"execution_count": null,
"metadata": {},
"outputs": [],
"source": [
"!pip install numpy pandas matplotlib torch torchvision pillow"
]
},
{
"cell_type": "markdown",
"metadata": {},
"source": [
"---\n",
"\n",
"## About the Dataset\n",
"\n",
"This project uses the Cats vs Dogs image dataset organized into separate directories for each class.\n",
"\n",
"### Classes\n",
"\n",
"- Cat\n",
"- Dog\n",
"\n",
"### Dataset Statistics\n",
"\n",
"| Dataset Split | Images |\n",
"|--------------|---------|\n",
"| Training Set | 14,998 |\n",
"| Validation Set | 4,999 |\n",
"| Test Set | 5,001 |\n",
"| **Total** | **24,998** |"
]
},
{
"cell_type": "code",
"execution_count": null,
"metadata": {},
"outputs": [],
"source": [
"import os\n",
"import pandas as pd\n",
"import numpy as np\n",
"import matplotlib.pyplot as plt\n",
"from PIL import Image\n",
"import torch\n",
"import torch.nn as nn\n",
"import torch.optim as optim\n",
"from torch.utils.data import DataLoader, random_split\n",
"from torchvision import datasets, transforms\n",
"\n",
"DATA_DIR = "dataset/PetImages"\n",
"print("Libraries successfully imported.")"
]
},
{
"cell_type": "markdown",
"metadata": {},
"source": [
"---\n",
"\n",
"## Data Preprocessing Pipeline\n",
"\n",
"### 1. Remove Corrupted Images\n",
"\n",
"Images are validated using the Pillow library. Files that cannot be opened are automatically removed. Removing corrupted files prevents DataLoader failures during training."
]
},
{
"cell_type": "code",
"execution_count": null,
"metadata": {},
"outputs": [],
"source": [
"def clean_dataset(data_dir):\n",
"    removed_count = 0\n",
"    for root, dirs, files in os.walk(data_dir):\n",
"        for file in files:\n",
"            if file.lower().endswith(('.jpg', '.jpeg', '.png')):\n",
"                path = os.path.join(root, file)\n",
"                try:\n",
"                    with Image.open(path) as img:\n",
"                        img.verify()\n",
"                except:\n",
"                    os.remove(path)\n",
"                    removed_count += 1\n",
"    print(f"Data cleaning complete. Removed {removed_count} corrupted images.")\n",
"\n",
"# clean_dataset(DATA_DIR)"
]
},
{
"cell_type": "markdown",
"metadata": {},
"source": [
"---\n",
"\n",
"### 2. Image Transformation\n",
"\n",
"All images are resized to 128x128 pixels and converted into tensors.\n",
"\n",
"text\n", "Original Image\n", "      ↓\n", "128 × 128 Resize\n", "      ↓\n", "Tensor Conversion\n", ""
]
},
{
"cell_type": "code",
"execution_count": null,
"metadata": {},
"outputs": [],
"source": [
"transform = transforms.Compose([\n",
"    transforms.Resize((128, 128)),\n",
"    transforms.ToTensor()\n",
"])\n",
"print("Transforms pipeline configured.")"
]
},
{
"cell_type": "markdown",
"metadata": {},
"source": [
"---\n",
"\n",
"### 3. Dataset Loading\n",
"\n",
"The dataset is loaded using TorchVision's ImageFolder class."
]
},
{
"cell_type": "code",
"execution_count": null,
"metadata": {},
"outputs": [],
"source": [
"# dataset = datasets.ImageFolder(DATA_DIR, transform=transform)\n",
"# print(f"Detected classes: {dataset.classes}")"
]
},
{
"cell_type": "markdown",
"metadata": {},
"source": [
"---\n",
"\n",
"### 4. Dataset Splitting\n",
"\n",
"The dataset is split into 60% Training, 20% Validation, and 20% Testing sets."
]
},
{
"cell_type": "code",
"execution_count": null,
"metadata": {},
"outputs": [],
"source": [
"total_images = 24998\n",
"train_size = 14998\n",
"val_size = 4999\n",
"test_size = 5001\n",
"\n",
"print(f"Splits configuration: Train={train_size}, Val={val_size}, Test={test_size}")\n",
"# train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])"
]
},
{
"cell_type": "markdown",
"metadata": {},
"source": [
"---\n",
"\n",
"## Data Loading\n",
"\n",
"PyTorch DataLoaders are used to batch and shuffle data using a batch size of 32."
]
},
{
"cell_type": "code",
"execution_count": null,
"metadata": {},
"outputs": [],
"source": [
"BATCH_SIZE = 32\n",
"\n",
"# train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)\n",
"# val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)\n",
"# test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)\n",
"print(f"DataLoaders configured with batch size: {BATCH_SIZE}")"
]
},
{
"cell_type": "markdown",
"metadata": {},
"source": [
"---\n",
"\n",
"## CNN Architecture\n",
"\n",
"The model consists of three convolutional blocks followed by a dense linear classifier.\n",
"\n",
"### Layer Processing Flow\n",
"text\n", "Input Image (3 × 128 × 128)\n", "        │\n", "        ├── Conv2D (3 → 16), ReLU, MaxPool\n", "        ├── Conv2D (16 → 32), ReLU, MaxPool\n", "        ├── Conv2D (32 → 64), ReLU, MaxPool\n", "        │\n", "     Flatten\n", "        │\n", "        ├── Linear (16384 → 128), ReLU\n", "        └── Linear (128 → 2)\n", ""
]
},
{
"cell_type": "code",
"execution_count": null,
"metadata": {},
"outputs": [],
"source": [
"class CatDogCNN(nn.Module):\n",
"    def **init**(self):\n",
"        super(CatDogCNN, self).**init**()\n",
"        self.features = nn.Sequential(\n",
"            nn.Conv2d(3, 16, kernel_size=3, padding=1),\n",
"            nn.ReLU(),\n",
"            nn.MaxPool2d(2, 2),\n",
"            \n",
"            nn.Conv2d(16, 32, kernel_size=3, padding=1),\n",
"            nn.ReLU(),\n",
"            nn.MaxPool2d(2, 2),\n",
"            \n",
"            nn.Conv2d(32, 64, kernel_size=3, padding=1),\n",
"            nn.ReLU(),\n",
"            nn.MaxPool2d(2, 2)\n",
"        )\n",
"        self.classifier = nn.Sequential(\n",
"            nn.Flatten(),\n",
"            nn.Linear(64 * 16 * 16, 128),\n",
"            nn.ReLU(),\n",
"            nn.Linear(128, 2)\n",
"        )\n",
"        \n",
"    def forward(self, x):\n",
"        x = self.features(x)\n",
"        x = self.classifier(x)\n",
"        return x\n",
"\n",
"model = CatDogCNN()\n",
"print(model)"
]
},
{
"cell_type": "markdown",
"metadata": {},
"source": [
"---\n",
"\n",
"## Device Configuration\n",
"\n",
"Training automatically falls back to Apple Silicon GPU hardware acceleration (MPS) when available."
]
},
{
"cell_type": "code",
"execution_count": null,
"metadata": {},
"outputs": [],
"source": [
"device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")\n",
"model.to(device)\n",
"print(f"Using target device execution backend: {device}")"
]
},
{
"cell_type": "markdown",
"metadata": {},
