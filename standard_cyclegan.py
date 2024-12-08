# -*- coding: utf-8 -*-
"""Standard_CycleGAN.ipynb

"""

from google.colab import drive
drive.mount('/content/drive')

import os
import random
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split, Subset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths to datasets
monet_train_path = "/content/drive/MyDrive/monet/train"
monet_test_path = "/content/drive/MyDrive/monet/test"

# Transformations
transform = transforms.Compose([
    transforms.Resize(128),
    #transforms.RandomCrop(128),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Custom Dataset Loader for Monet
class MonetDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        """
        Args:
            folder_path (string): Path to the folder (train or test) with images.
            transform (callable, optional): Transformations to apply to images.
        """
        self.image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Convert to RGB
        if self.transform:
            image = self.transform(image)
        return image, 0  # No label required, so returning 0 as placeholder

# Load Monet Dataset
monet_train_dataset = MonetDataset(folder_path=monet_train_path, transform=transform)
monet_test_dataset = MonetDataset(folder_path=monet_test_path, transform=transform)

# CIFAR-10 Dataset
cifar10_dataset = datasets.CIFAR10(root="/content/drive/MyDrive/data", train=True, download=True, transform=transform)

# Randomly sample 1000 images from CIFAR-10
sample_size_cifar10 = 1000
indices = random.sample(range(len(cifar10_dataset)), sample_size_cifar10)
cifar10_subset = Subset(cifar10_dataset, indices)

# Split CIFAR-10 into train and test (80% train, 20% test)
train_size = int(0.8 * len(cifar10_subset))
test_size = len(cifar10_subset) - train_size
cifar10_train, cifar10_test = random_split(cifar10_subset, [train_size, test_size])

# DataLoaders
batch_size = 8
train_loader_A = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
test_loader_A = DataLoader(cifar10_test, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

train_loader_B = DataLoader(monet_train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
test_loader_B = DataLoader(monet_test_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

# Visualization Functions
def denormalize(tensor):
    tensor = tensor * 0.5 + 0.5  # Undo normalization
    return tensor.clamp(0, 1)

def show_images(images, title):
    images = images.cpu().numpy().transpose(0, 2, 3, 1)
    plt.figure(figsize=(12, 6))
    plt.imshow(np.concatenate(images, axis=1))
    plt.title(title)
    plt.axis('off')
    plt.show()

# Visualize CIFAR-10 (Domain A)
batch_A, _ = next(iter(train_loader_A))
show_images(denormalize(batch_A[:8]), title="Sample Images from CIFAR-10 (Domain A)")

# Visualize Monet (Domain B)
batch_B, _ = next(iter(train_loader_B))
show_images(denormalize(batch_B[:8]), title="Sample Images from Monet (Domain B)")

import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, input_channels=3, num_residuals=9):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.downsampling = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(num_residuals)]
        )
        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.output_layer = nn.Sequential(
            nn.Conv2d(64, input_channels, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.downsampling(x)
        x = self.residual_blocks(x)
        x = self.upsampling(x)
        return self.output_layer(x)

class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)

import torch.optim as optim
import itertools


# Loss functions
adversarial_loss = nn.MSELoss().to(device)
cycle_consistency_loss = nn.L1Loss().to(device)
identity_loss = nn.L1Loss().to(device)

# Initialize models
G_AtoB = Generator().to(device)
G_BtoA = Generator().to(device)
D_A = Discriminator().to(device)
D_B = Discriminator().to(device)

# Optimizers
lr = 0.0001
beta1 = 0.5
optimizer_G = optim.Adam(itertools.chain(G_AtoB.parameters(), G_BtoA.parameters()), lr=lr, betas=(beta1, 0.999))
optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=(beta1, 0.999))

# Training loop
epochs = 30
lambda_cycle = 15.0
lambda_identity = 10.0

for epoch in range(epochs):
    for i, (data_A, data_B) in enumerate(zip(train_loader_A, train_loader_B)):
        real_A = data_A[0].to(device)
        real_B = data_B[0].to(device)

        # Train Generators
        optimizer_G.zero_grad()
        fake_B = G_AtoB(real_A)
        fake_A = G_BtoA(real_B)
        recovered_A = G_BtoA(fake_B)
        recovered_B = G_AtoB(fake_A)

        # Loss
        id_loss = identity_loss(G_BtoA(real_A), real_A) * lambda_identity + \
                  identity_loss(G_AtoB(real_B), real_B) * lambda_identity
        gan_loss = adversarial_loss(D_B(fake_B), torch.ones_like(D_B(fake_B))) + \
                   adversarial_loss(D_A(fake_A), torch.ones_like(D_A(fake_A)))
        cycle_loss = cycle_consistency_loss(recovered_A, real_A) * lambda_cycle + \
                     cycle_consistency_loss(recovered_B, real_B) * lambda_cycle
        loss_G = id_loss + gan_loss + cycle_loss
        loss_G.backward()
        optimizer_G.step()

        # Train Discriminators
        optimizer_D_A.zero_grad()
        loss_D_A = adversarial_loss(D_A(real_A), torch.ones_like(D_A(real_A))) + \
                   adversarial_loss(D_A(fake_A.detach()), torch.zeros_like(D_A(fake_A.detach())))
        loss_D_A.backward()
        optimizer_D_A.step()

        optimizer_D_B.zero_grad()
        loss_D_B = adversarial_loss(D_B(real_B), torch.ones_like(D_B(real_B))) + \
                   adversarial_loss(D_B(fake_B.detach()), torch.zeros_like(D_B(fake_B.detach())))
        loss_D_B.backward()
        optimizer_D_B.step()

    print(f"Epoch [{epoch + 1}/{epochs}] - Loss G: {loss_G.item():.4f}, Loss D_A: {loss_D_A.item():.4f}, Loss D_B: {loss_D_B.item():.4f}")

import os
import shutil

# Path to the folder containing files
folder_path = "/content/content/drive/MyDrive/fake_images"

# Delete all files in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)  # Delete the file
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)  # Delete the folder
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")

print(f"All files in {folder_path} have been deleted.")

import os
import torch
from torchvision.utils import save_image
from PIL import Image

# Updated function to save images from a DataLoader
def save_images(dataloader, output_dir, prefix="real", max_images=80):
    """
    Saves images from a dataloader to the specified directory.

    Args:
        dataloader (DataLoader): Dataloader containing images to save.
        output_dir (str): Directory to save the images.
        prefix (str): Prefix for saved image filenames.
        max_images (int): Maximum number of images to save.
    """
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
    saved_count = 0  # Track how many images have been saved

    for i, (images, _) in enumerate(dataloader):
        print(f"Processing batch {i}, batch size: {images.size(0)}")  # Debugging batch info
        for j in range(images.size(0)):  # Iterate over images in the batch
            if saved_count >= max_images:
                print(f"Saved {saved_count} images to {output_dir}.")
                return  # Stop once max_images is reached
            image_path = os.path.join(output_dir, f"{prefix}_{saved_count}.png")
            try:
                save_image(images[j], image_path)
                print(f"Saved image: {image_path}")  # Confirm successful save
            except Exception as e:
                print(f"Error saving image {image_path}: {e}")
            saved_count += 1

    print(f"Saved {saved_count} images to {output_dir}.")

# Check the DataLoader
def check_dataloader(dataloader, name="DataLoader"):
    """
    Debug function to check the dataloader and display the first image.
    """
    print(f"Checking {name}...")
    try:
        images, _ = next(iter(dataloader))
        print(f"Batch shape: {images.shape}")
        # Save the first image as a test
        test_image_path = "./test_image.png"
        save_image(images[0], test_image_path)
        print(f"Test image saved successfully at {test_image_path}")
    except Exception as e:
        print(f"Error with {name}: {e}")

# Paths and Variables
real_images_dir = "./content/drive/MyDrive/real_images"
os.makedirs(real_images_dir, exist_ok=True)

# Step 1: Validate DataLoader
print("Validating test_loader_B...")
check_dataloader(test_loader_B, name="test_loader_B")

# Step 2: Save Images
print("Saving images from test_loader_B...")
save_images(test_loader_B, real_images_dir, prefix="real_monet", max_images=80)

def generate_and_save_images(generator, dataloader, output_dir, prefix="fake", count=10):
    os.makedirs(output_dir, exist_ok=True)
    generator.eval()
    device = next(generator.parameters()).device
    for i, (real_images, _) in enumerate(dataloader):
        if i >= count:
            break
        real_images = real_images.to(device)
        with torch.no_grad():
            fake_images = generator(real_images)
        for j in range(fake_images.size(0)):
            save_image(fake_images[j], os.path.join(output_dir, f"{prefix}_{i * len(real_images) + j}.png"))

# Save Generated Images
generate_and_save_images(G_AtoB, test_loader_A, "./content/drive/MyDrive/fake_images", prefix="fake_monet")

!pip install torch-fidelity

from torch_fidelity import calculate_metrics

def compute_fid(real_dir, fake_dir):
    metrics = calculate_metrics(
        input1=real_dir,
        input2=fake_dir,
        fid=True,  # Enable FID calculation
        cuda=True  # Use GPU if available
    )
    print(f"FID: {metrics['frechet_inception_distance']}")
    return metrics['frechet_inception_distance']

# Compute FID
real_dir = "./content/drive/MyDrive/real_images"
fake_dir = "./content/drive/MyDrive/fake_images"
fid_score = compute_fid(real_dir, fake_dir)

import os
from torchvision.utils import save_image
from torch_fidelity import calculate_metrics

def compute_inception_score(fake_dir):
    """
    Compute the Inception Score (IS) for generated images using torch_fidelity.

    Args:
        fake_dir (str): Directory containing the generated (fake) images.

    Returns:
        float: The calculated Inception Score.
    """
    metrics = calculate_metrics(
        input1=fake_dir,
        cuda=torch.cuda.is_available(),
        isc=True,  # Enables Inception Score calculation
        fid=False  # Disables FID calculation
    )
    inception_score = metrics['inception_score_mean']
    print(f"Inception Score: {inception_score}")
    return inception_score


# Example usage for your test_loader_B (fake images)
fake_images_dir = "./content/drive/MyDrive/fake_images"
os.makedirs(fake_images_dir, exist_ok=True)

# Save generated images from your DataLoader
print("Saving fake images...")
save_images(test_loader_B, fake_images_dir, prefix="fake_monet", max_images=100)

# Compute Inception Score
print("Computing Inception Score...")
compute_inception_score(fake_images_dir)

G_AtoB.eval()
G_BtoA.eval()

# Generate images
with torch.no_grad():
    test_batch_A, _ = next(iter(test_loader_A))
    test_batch_B, _ = next(iter(test_loader_B))

    real_A = test_batch_A.to(device)
    real_B = test_batch_B.to(device)

    fake_B = G_AtoB(real_A)
    fake_A = G_BtoA(real_B)

# Visualize generated images
show_images(denormalize(real_A[:10]), title="Real CIFAR-10 (Domain A)")
show_images(denormalize(fake_B[:10]), title="Generated Monet (Domain B)")
#show_images(denormalize(real_B[:8]), title="Real Monet (Domain B)")
#show_images(denormalize(fake_A[:8]), title="Generated CIFAR-10 (Domain A)")