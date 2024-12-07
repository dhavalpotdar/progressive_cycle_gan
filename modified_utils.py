import os
import torch
from torch_fidelity import calculate_metrics

def compute_fid(real_dir, fake_dir):
    """
    Compute Fréchet Inception Distance (FID) between real and fake image directories.

    Args:
        real_dir (str): Path to the directory containing real images.
        fake_dir (str): Path to the directory containing fake images.

    Returns:
        float: FID score.
    """
    metrics = calculate_metrics(input1=real_dir, input2=fake_dir, fid=True, verbose=True)
    return metrics["frechet_inception_distance"]

def compute_inception_score(fake_dir):
    """
    Compute Inception Score (IS) for fake image directory.

    Args:
        fake_dir (str): Path to the directory containing fake images.

    Returns:
        float: Inception score mean.
    """
    metrics = calculate_metrics(input1=fake_dir, inception_score=True, verbose=True)
    return metrics["inception_score_mean"]

def save_checkpoint(
    path, gen_A2B, gen_B2A, disc_A, disc_B, opt_gen, opt_disc, step, epoch
):
    checkpoint = {
        "gen_A2B_state_dict": gen_A2B.state_dict(),
        "gen_B2A_state_dict": gen_B2A.state_dict(),
        "disc_A_state_dict": disc_A.state_dict(),
        "disc_B_state_dict": disc_B.state_dict(),
        "opt_gen_state_dict": opt_gen.state_dict(),
        "opt_disc_state_dict": opt_disc.state_dict(),
        "step": step,
        "epoch": epoch,
    }

    torch.save(
        checkpoint,
        os.path.join(path, f"checkpoint_step_{step}_epoch_{epoch}.pth"),
    )

    print(f"Checkpoint saved at step {step}, epoch {epoch}.")


def load_checkpoint(filepath, gen_A2B, gen_B2A, disc_A, disc_B, opt_gen, opt_disc):
    if not os.path.exists(filepath):
        print(f"No checkpoint found at {filepath}.")
        return 0, 0  # Return default step and epoch

    checkpoint = torch.load(filepath)
    gen_A2B.load_state_dict(checkpoint["gen_A2B_state_dict"])
    gen_B2A.load_state_dict(checkpoint["gen_B2A_state_dict"])
    disc_A.load_state_dict(checkpoint["disc_A_state_dict"])
    disc_B.load_state_dict(checkpoint["disc_B_state_dict"])
    opt_gen.load_state_dict(checkpoint["opt_gen_state_dict"])
    opt_disc.load_state_dict(checkpoint["opt_disc_state_dict"])
    step = checkpoint["step"]
    epoch = checkpoint["epoch"]
    print(f"Checkpoint loaded from step {step}, epoch {epoch}.")
    return step, epoch


def load_genA2B(filepath, generator):

    checkpoint = torch.load(filepath)
    generator.load_state_dict(checkpoint["gen_A2B_state_dict"])

    step = checkpoint["step"]
    epoch = checkpoint["epoch"]
    print(f"Checkpoint loaded from step {step}, epoch {epoch}.")
    return generator
