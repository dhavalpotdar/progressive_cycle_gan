import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from input_datasets.celeba import CelebADataset

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # Nor
    ]
)

# Load CIFAR10 test dataset ###################################################


# Test on CIFAR10
def test_on_cifar10(generator, num_steps, img_count=10, fake_dir="eval_images/fake"):

    cifar10_test = datasets.CIFAR10(
        root="data", train=False, download=True, transform=transform
    )
    dataloader = DataLoader(cifar10_test, batch_size=1, shuffle=False)

    generator.eval()
    device = next(generator.parameters()).device

    for i, (real_A, _) in enumerate(dataloader):
        if i >= img_count:
            break

        real_A = real_A.to(device)
        with torch.no_grad():
            fake_B = generator(real_A, alpha=1.0, steps=num_steps - 1)
        
        # Save the generated fake image
        save_image(fake_B * 0.5 + 0.5, os.path.join(fake_dir, f"fake_{i}.png"))

        # Optionally plot the results
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(real_A[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
        axes[0].set_title("Original CIFAR10")
        axes[0].axis("off")

        axes[1].imshow(fake_B[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
        axes[1].set_title("Monet Style")
        axes[1].axis("off")

        plt.show()


# Define paths and transformations for CELEBA #######################################


# Test on CELEBA
def test_on_celeba(generator, num_steps, img_count=10, fake_dir="eval_images/fake"):
    """
    Test the generator on CELEBA test images and save/display results.

    Args:
        generator (nn.Module): The trained generator model.
        dataloader (DataLoader): DataLoader for the test dataset.
        output_dir (str): Directory to save the test images.
        img_count (int): Number of test images to process.
    """
    celeba_img_dir = "data/CELEBA/img_align_celeba/img_align_celeba"
    celeba_partition_file = "data/CELEBA/list_eval_partition.csv"

    # Load CELEBA test dataset (partition 2)
    celeba_test_dataset = CelebADataset(
        img_dir=celeba_img_dir,
        partition_file=celeba_partition_file,
        partition=2,  # 2 = test set
        transform=transform,
    )
    dataloader = DataLoader(celeba_test_dataset, batch_size=1, shuffle=False)

    generator.eval()
    device = next(generator.parameters()).device

    for i, (real_A, _) in enumerate(dataloader):
        if i >= img_count:
            break

        real_A = real_A.to(device)
        with torch.no_grad():
            fake_B = generator(real_A, alpha=1.0, steps=num_steps - 1)

        # Save the generated fake image
        save_image(fake_B * 0.5 + 0.5, os.path.join(fake_dir, f"fake_{i}.png"))
        
        # Optionally plot the results
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(real_A[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
        axes[0].set_title("Original CELEBA")
        axes[0].axis("off")

        axes[1].imshow(fake_B[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
        axes[1].set_title("Van Gogh Style")
        axes[1].axis("off")

        plt.show()
