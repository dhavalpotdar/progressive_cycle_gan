import os
import torch
from PIL import Image


# Custom Dataset for Van Gogh
class VanGoghDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform=None):
        """
        Custom dataset for Van Gogh images.

        Args:
            folder_path (str): Path to the directory containing Van Gogh images.
            transform (callable, optional): Transform to apply to the images.
        """
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
# vangogh_train_path = "data/vangogh/train"

# # Transformation for Van Gogh images
# vangogh_transform = transforms.Compose(
#     [
#         transforms.Resize(128),  # Adjust size as needed
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
#         ),  # Normalize to [-1, 1]
#     ]
# )

# # Create Van Gogh training dataset
# vangogh_train_dataset = VanGoghDataset(
#     folder_path=vangogh_train_path, transform=vangogh_transform
# )

# # Create DataLoader for Van Gogh
# vangogh_train_loader = DataLoader(
#     vangogh_train_dataset, batch_size=8, shuffle=True, num_workers=0
# )
