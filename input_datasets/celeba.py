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


# Test on CELEBA
def test_on_celeba(generator, dataloader, output_dir, num_steps, img_count=10):
    """
    Test the generator on CELEBA test images and save/display results.

    Args:
        generator (nn.Module): The trained generator model.
        dataloader (DataLoader): DataLoader for the test dataset.
        output_dir (str): Directory to save the test images.
        img_count (int): Number of test images to process.
    """
    os.makedirs(output_dir, exist_ok=True)
    generator.eval()
    device = next(generator.parameters()).device

    for i, (real_A, _) in enumerate(dataloader):
        if i >= img_count:
            break

        real_A = real_A.to(device)
        with torch.no_grad():
            fake_B = generator(real_A, alpha=1.0, steps=num_steps - 1)

        # Save the results
        save_image(
            fake_B * 0.5 + 0.5, os.path.join(output_dir, f"vangogh_style_{i}.png")
        )  # Rescale to [0, 1]
        save_image(real_A * 0.5 + 0.5, os.path.join(output_dir, f"original_{i}.png"))

        # Optionally plot the results
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(real_A[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
        axes[0].set_title("Original CELEBA")
        axes[0].axis("off")

        axes[1].imshow(fake_B[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
        axes[1].set_title("Van Gogh Style")
        axes[1].axis("off")

        plt.show()
