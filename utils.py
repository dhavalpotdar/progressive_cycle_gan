import os
import torch


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
