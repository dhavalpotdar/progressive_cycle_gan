from random import random
import torch
from torchvision import transforms, datasets


class CustomCIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, indices, transform=None):
        """
        Custom dataset class for CIFAR10 with random sampling.
        Args:
            original_dataset (torch.utils.data.Dataset): The original CIFAR10 dataset.
            indices (list): List of indices for random sampling.
            transform (callable, optional): Transform to apply to the images.
        """
        self.original_dataset = original_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        img, label = self.original_dataset[original_idx]
        if self.transform:
            img = self.transform(img)
        return img, label


# EXAMPLE USAGE ############################################################################
# transform = transforms.Compose([transforms.ToTensor()])

# # Load the full CIFAR10 dataset
# cifar10_full_dataset = datasets.CIFAR10(
#     root="data", train=True, download=True, transform=None
# )

# # Randomly sample 1000 images
# sample_size_cifar10 = 1000
# indices = random.sample(range(len(cifar10_full_dataset)), sample_size_cifar10)

# # Create a custom dataset with the sampled indices
# cifar10_train_dataset = CustomCIFAR10Dataset(
#     original_dataset=cifar10_full_dataset, indices=indices, transform=transform
# )
