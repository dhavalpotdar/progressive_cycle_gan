# P1: Progressive Growing CycleGAN for High-Resolution Image Translation

This project integrates CycleGAN with Progressive Growing GAN (ProGAN) to perform high-resolution image-to-image translation using unpaired datasets. The goal is to progressively increase image resolution during training to enhance stability and image quality, enabling the model to translate styles between domains effectively.

---

## File Descriptions

### Core Project Files

1. **`utils.py`**  
   Utility functions for:
   - Computing **Fréchet Inception Distance (FID)** and **Inception Score (IS)**.
   - Saving and loading checkpoints for the generator and discriminator models.
   - Loading the generator model (`load_genA2B`) for evaluation purposes.

2. **`train.py`**  
   - Main training script for the Progressive Growing CycleGAN.  
   - Supports various source (e.g., CelebA, CIFAR-10) and style datasets (e.g., Monet, Van Gogh).  
   - Implements dynamic resolution adjustment, fade-in, and custom loss functions.  
   - Outputs training logs, intermediate results, and checkpoint files.

3. **`testloaders.py`**  
   - Scripts for testing the trained model on datasets like CIFAR-10 and CelebA.  
   - Generates styled images (e.g., Monet-style, Van Gogh-style) and saves them for evaluation.  
   - Includes visualization utilities for comparing original and translated images.

4. **`modules.py`**  
   Core building blocks for the Progressive CycleGAN:
   - Weighted-Scaled Convolution layers (`WSConv2d`) for stability.
   - Pixel Normalization for consistent training.
   - Residual blocks and convolutional blocks used in the generator and discriminator.

5. **`losses.py`**  
   - Custom loss functions for CycleGAN:
     - **Cycle Loss:** Ensures the model learns consistent transformations.
     - **Least Squares GAN Loss (LSGAN):** Stabilizes GAN training.

6. **`generator.py`**  
   - Defines the Progressive Growing CycleGAN generator architecture.
   - Includes:
     - Progressive layer addition with fade-in.
     - Residual blocks for enhanced learning of transformations.

7. **`discriminator.py`**  
   - Implements the CycleGAN discriminator, supporting progressive resizing.
   - Includes:
     - Minibatch Standard Deviation for diversity.
     - Weighted-Scaled Convolutions for stable training.

---

### Dataset Scripts (Inside `input_datasets/`)

1. **`celeba.py`**  
   - Custom dataset loader for the CelebA dataset.  
   - Supports train/test partitioning and preprocessing transformations.

2. **`cifar10.py`**  
   - Custom dataset class for CIFAR-10 with random sampling support.  
   - Preprocesses images for compatibility with CycleGAN training.

3. **`monet.py`**  
   - Custom dataset loader for Monet paintings.  
   - Prepares training and test datasets for style translation tasks.

4. **`vangogh.py`**  
   - Custom dataset loader for Van Gogh paintings.  
   - Supports training with public domain Van Gogh images.

---

### Configuration File

- **`config.ini`**  
   - Centralized configuration for training parameters, dataset paths, and model settings.  
   - Customizable fields for:
     - Training hyperparameters (e.g., image size, batch size, learning rate).  
     - Dataset selection (e.g., CelebA, CIFAR-10, Monet, Van Gogh).  
     - Output directory for generated images and checkpoints.

---

## Usage

check `Example_Usage.ipynb` for Usage.

# Evaluation

To evaluate the trained model and generate stylized images, use the following command:

```
python testloaders.py
```

# Metrics

Compute FID and Inception Scores using the `utils.py` functions:

```
from utils import compute_fid, compute_inception_score
fid_score = compute_fid(real_dir="data/vangogh/train", fake_dir="eval_images/fake")
inception_score = compute_inception_score(fake_dir="eval_images/fake")

```

# Results

The project showcases:

- **CelebA to Van Gogh** style translation.  
- **CIFAR-10 to Monet** style translation.  

## Key Metrics

- **Fréchet Inception Distance (FID)**  
- **Inception Score (IS)**  

Refer to the Project Report for detailed results and analysis.


