import os
import configparser
import itertools
import math
import random
from datetime import datetime

import torch
import torch.optim as optim
from torchvision.utils import save_image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F

from generator import GeneratorCycleGAN
from discriminator import DiscriminatorCycleGAN
from utils import save_checkpoint
from losses import CycleLoss, LSGANLoss
from errors import InvalidDatasetError

config = configparser.ConfigParser()
config.read("config.ini")

# Training Hyperparameters
START_TRAIN_IMG_SIZE = int(config["TRAINING"]["start_train_img_size"])
TARGET_IMG_SIZE = int(config["TRAINING"]["target_img_size"])
Z_DIM = int(config["TRAINING"]["z_dim"])
IN_CHANNELS = int(config["TRAINING"]["in_channels"])
IMG_CHANNELS = int(config["TRAINING"]["img_channels"])
BATCH_SIZE = int(config["TRAINING"]["batch_size"])
LEARNING_RATE = float(config["TRAINING"]["learning_rate"])
EPOCHS = int(config["TRAINING"]["num_epochs"])
DEVICE = eval(config["TRAINING"]["device"])  # Evaluate the device logic
OUTPUT_DIR = config["OUTPUTS"]["output_dir"]

# Calculate NUM_STEPS if set to 'auto'
if str(config["TRAINING"]["num_steps"]) == "auto":
    NUM_STEPS = int(math.log2(TARGET_IMG_SIZE / START_TRAIN_IMG_SIZE)) + 1
else:
    NUM_STEPS = int(config["TRAINING"]["num_steps"])

# Dataset Names
SOURCE_DATASET = config["DATASETS"]["source_dataset"]
STYLE_DATASET = config["DATASETS"]["style_dataset"]

# Print loaded variables (optional, for debugging)
print(f"START_TRAIN_IMG_SIZE: {START_TRAIN_IMG_SIZE}")
print(f"TARGET_IMG_SIZE: {TARGET_IMG_SIZE}")
print(f"Z_DIM: {Z_DIM}")
print(f"IN_CHANNELS: {IN_CHANNELS}")
print(f"IMG_CHANNELS: {IMG_CHANNELS}")
print(f"BATCH_SIZE: {BATCH_SIZE}")
print(f"LEARNING_RATE: {LEARNING_RATE}")
print(f"EPOCHS: {EPOCHS}")
print(f"DEVICE: {DEVICE}")
print(f"NUM_STEPS: {NUM_STEPS}")
print(f"SOURCE_DATASET: {SOURCE_DATASET}")
print(f"STYLE_DATASET: {STYLE_DATASET}")
print(f"OUTPUT_DIR: {OUTPUT_DIR}")


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
        ),  # Normalize to [-1, 1]
    ]
)

# Load Datasets
# Source:
if SOURCE_DATASET == "CIFAR10":
    from input_datasets.cifar10 import CustomCIFAR10Dataset

    # Load the full CIFAR10 dataset
    cifar10_full_dataset = datasets.CIFAR10(
        root="data", train=True, download=True, transform=None
    )

    # Randomly sample 1000 images
    sample_size_cifar10 = 1000
    indices = random.sample(range(len(cifar10_full_dataset)), sample_size_cifar10)

    cifar10_train_dataset = CustomCIFAR10Dataset(
        original_dataset=cifar10_full_dataset, indices=indices, transform=transform
    )

    train_loader_A = DataLoader(
        cifar10_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )


elif SOURCE_DATASET == "CELEBA":
    from input_datasets.celeba import CelebADataset

    celeba_img_dir = "data/CELEBA/img_align_celeba/img_align_celeba"
    celeba_partition_file = "data/CELEBA/list_eval_partition.csv"

    # Create CELEBA training dataset
    celeba_train_dataset = CelebADataset(
        img_dir=celeba_img_dir,
        partition_file=celeba_partition_file,
        partition=0,  # 0 = train
        transform=transform,
    )

    # Load CELEBA test dataset (partition 2)
    celeba_test_dataset = CelebADataset(
        img_dir=celeba_img_dir,
        partition_file=celeba_partition_file,
        partition=2,  # 2 = test set
        transform=transform,
    )

    # Create DataLoader for CELEBA
    train_loader_A = DataLoader(
        celeba_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )

    test_loader_A = DataLoader(celeba_test_dataset, batch_size=1, shuffle=False)


elif SOURCE_DATASET == "LSUN":
    raise NotImplementedError

else:
    raise InvalidDatasetError(
        "Invalid Source Dataset. Please set the source dataset in config.ini to either of CIFAR10, LSUN or CELEBA."
    )

# Style:
if STYLE_DATASET == "Vangogh":
    from input_datasets.vangogh import VanGoghDataset

    vangogh_train_path = "data/vangogh/train"

    vangogh_train_dataset = VanGoghDataset(
        folder_path=vangogh_train_path, transform=transform
    )

    train_loader_B = DataLoader(
        vangogh_train_dataset, batch_size=8, shuffle=True, num_workers=0
    )


elif STYLE_DATASET == "Monet":
    from input_datasets.monet import MonetDataset

    # Paths to datasets
    monet_train_path = "data/monet/train"
    monet_test_path = "data/monet/test"

    # Load Monet Dataset
    monet_train_dataset = MonetDataset(
        folder_path=monet_train_path, transform=transform
    )

    train_loader_B = DataLoader(
        monet_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )

else:
    raise InvalidDatasetError(
        "Invalid Style Dataset. Please set the source dataset in config.ini to either of Vangogh or Monet."
    )


def train_prog_cycle_gan():
    # Instantiate models
    gen_A2B = GeneratorCycleGAN(in_channels=256, img_channels=3).to(DEVICE)
    gen_B2A = GeneratorCycleGAN(in_channels=256, img_channels=3).to(DEVICE)
    disc_A = DiscriminatorCycleGAN(in_channels=256, img_channels=3).to(DEVICE)
    disc_B = DiscriminatorCycleGAN(in_channels=256, img_channels=3).to(DEVICE)

    # Optimizers
    opt_gen = optim.Adam(
        itertools.chain(gen_A2B.parameters(), gen_B2A.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    opt_disc = optim.Adam(
        itertools.chain(disc_A.parameters(), disc_B.parameters()),
        lr=LEARNING_RATE * 0.5,
        betas=(0.5, 0.999),
    )

    # Losses
    cycle_loss = CycleLoss()
    adversarial_loss = LSGANLoss()

    # create a directory for current run
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    WORKING_DIR = os.path.join(
        OUTPUT_DIR, SOURCE_DATASET + " to " + STYLE_DATASET + " " + timestamp
    )
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Training Loop
    for step in range(0, NUM_STEPS):
        img_size = START_TRAIN_IMG_SIZE * (2**step)
        alpha = 1e-5
        print(f"Training at image size {img_size}x{img_size}")

        train_loader_A.dataset.transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
        train_loader_B.dataset.transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        for epoch in range(0, EPOCHS):  # Adjust epochs as needed
            for batch_idx, ((real_A, _), (real_B, _)) in enumerate(
                zip(train_loader_A, train_loader_B)
            ):
                real_A = real_A.to(DEVICE)
                real_B = real_B.to(DEVICE)

                # Train Discriminators
                fake_B = gen_A2B(real_A, alpha, step)
                fake_A = gen_B2A(real_B, alpha, step)

                # Discriminator A
                disc_real_A = disc_A(real_A, alpha, step)
                disc_fake_A = disc_A(fake_A.detach(), alpha, step)
                loss_disc_A = (
                    adversarial_loss(disc_real_A, target_is_real=True)
                    + adversarial_loss(disc_fake_A, target_is_real=False)
                ) / 2

                # Discriminator B
                disc_real_B = disc_B(real_B, alpha, step)
                disc_fake_B = disc_B(fake_B.detach(), alpha, step)
                loss_disc_B = (
                    adversarial_loss(disc_real_B, target_is_real=True)
                    + adversarial_loss(disc_fake_B, target_is_real=False)
                ) / 2

                loss_disc = loss_disc_A + loss_disc_B
                opt_disc.zero_grad()
                loss_disc.backward()
                opt_disc.step()

                # Train Generators
                fake_B = gen_A2B(real_A, alpha, step)
                fake_A = gen_B2A(real_B, alpha, step)

                disc_fake_A = disc_A(fake_A, alpha, step)
                disc_fake_B = disc_B(fake_B, alpha, step)

                loss_gen_A2B = adversarial_loss(disc_fake_B, target_is_real=True)
                loss_gen_B2A = adversarial_loss(disc_fake_A, target_is_real=True)

                # Cycle Consistency Loss
                recon_A = gen_B2A(fake_B, alpha, step)
                recon_B = gen_A2B(fake_A, alpha, step)

                # Ensure reconstructions match real image sizes
                if recon_A.shape != real_A.shape:
                    recon_A = F.interpolate(
                        recon_A, size=real_A.shape[2:], mode="bilinear"
                    )
                if recon_B.shape != real_B.shape:
                    recon_B = F.interpolate(
                        recon_B, size=real_B.shape[2:], mode="bilinear"
                    )

                loss_cycle_A = cycle_loss(recon_A, real_A)
                loss_cycle_B = cycle_loss(recon_B, real_B)

                # Identity Loss
                id_A = gen_B2A(real_A, alpha, step)  # Preserve real_A's style
                id_B = gen_A2B(real_B, alpha, step)  # Preserve real_B's style

                # Ensure identity outputs match real image sizes
                if id_A.shape != real_A.shape:
                    id_A = F.interpolate(
                        id_A,
                        size=real_A.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                if id_B.shape != real_B.shape:
                    id_B = F.interpolate(
                        id_B,
                        size=real_B.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                # Calculate Identity Loss
                loss_identity_A = cycle_loss(id_A, real_A)
                loss_identity_B = cycle_loss(id_B, real_B)

                loss_gen = (
                    loss_gen_A2B
                    + loss_gen_B2A
                    + 20
                    * (loss_cycle_A + loss_cycle_B)  # Increase cycle consistency weight
                    + 5
                    * (
                        loss_identity_A + loss_identity_B
                    )  # Increase identity loss weight
                )

                opt_gen.zero_grad()
                loss_gen.backward()
                opt_gen.step()

                # Update alpha for fade-in
                alpha += 1e-3
                alpha = min(alpha, 1.0)

                if batch_idx % 200 == 0:
                    print(
                        f"Epoch [{epoch}/{100}] Batch {batch_idx} "
                        f"Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}, alpha: {alpha:.4f}"
                    )

                    # print("-" * 100)
                    # print(
                    #     f"Discriminator Loss A: {loss_disc_A.item()}, Discriminator Loss B: {loss_disc_B.item()}"
                    # )
                    # print("-" * 100 + "\n")

                    # print("-" * 100)
                    # print(
                    #     f"Identity Loss A: {loss_identity_A.item()}, Identity Loss B: {loss_identity_B.item()}"
                    # )
                    # print("-" * 100 + "\n")

                    # print("-" * 100)
                    # print(
                    #     f"Cycle Loss A: {loss_cycle_A.item()}, Cycle Loss B: {loss_cycle_B.item()}"
                    # )
                    # print("-" * 100 + "\n")

                    save_image(
                        fake_B * 0.5 + 0.5,
                        os.path.join(
                            WORKING_DIR, f"fake_B_step_{step}_epoch_{epoch}.png"
                        ),
                    )
                    save_image(
                        fake_A * 0.5 + 0.5,
                        os.path.join(
                            WORKING_DIR, f"fake_A_step_{step}_epoch_{epoch}.png"
                        ),
                    )
                    save_image(
                        recon_A * 0.5 + 0.5,
                        os.path.join(
                            WORKING_DIR, f"recon_A_step_{step}_epoch_{epoch}.png"
                        ),
                    )
                    save_image(
                        recon_B * 0.5 + 0.5,
                        os.path.join(
                            WORKING_DIR, f"recon_B_step_{step}_epoch_{epoch}.png"
                        ),
                    )

            # Save checkpoint at the end of each epoch
            save_checkpoint(
                WORKING_DIR,
                gen_A2B,
                gen_B2A,
                disc_A,
                disc_B,
                opt_gen,
                opt_disc,
                step,
                epoch,
            )

        # Reset epoch counter after each step
        last_epoch = 0

    return WORKING_DIR, gen_A2B, gen_B2A
