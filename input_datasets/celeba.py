import os
from PIL import Image
import pandas as pd
import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt


# Custom Dataset for CELEBA
class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, partition_file, partition, transform=None):
        """
        Custom dataset for CELEBA images.

        Args:
            img_dir (str): Directory containing CELEBA images.
            partition_file (str): Path to the CSV file containing image partitions.
            partition (int): Partition type (0 = train, 1 = eval, 2 = test).
            transform (callable, optional): Transform to apply to the images.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.data = pd.read_csv(partition_file)
        self.data = self.data[self.data["partition"] == partition]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0  # No labels required


# EXAMPLE USAGE ############################################################################
# celeba_img_dir = "data/CELEBA/img_align_celeba/img_align_celeba"
# celeba_partition_file = "data/CELEBA/list_eval_partition.csv"

# # Transformation for CELEBA images
# celeba_transform = transforms.Compose(
#     [
#         transforms.Resize(128),  # Adjust size as needed
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
#         ),  # Normalize to [-1, 1]
#     ]
# )

# # Create CELEBA training dataset
# celeba_train_dataset = CelebADataset(
#     img_dir=celeba_img_dir,
#     partition_file=celeba_partition_file,
#     partition=0,  # 0 = train
#     transform=celeba_transform,
# )

# # Create DataLoader for CELEBA
# celeba_train_loader = DataLoader(
#     celeba_train_dataset, batch_size=8, shuffle=True, num_workers=0
# )
