import os
import torch
from PIL import Image


class MonetDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform=None):
        self.image_paths = [
            os.path.join(folder_path, img)
            for img in os.listdir(folder_path)
            if img.endswith((".png", ".jpg", ".jpeg"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0  # No labels required


# EXAMPLE USAGE ############################################################################
# # Paths to datasets
# monet_train_path = "data/monet/train"
# monet_test_path = "data/monet/test"

# # Load Monet Dataset
# monet_train_dataset = MonetDataset(folder_path=monet_train_path, transform=transform)
# monet_test_dataset = MonetDataset(folder_path=monet_test_path, transform=transform)
