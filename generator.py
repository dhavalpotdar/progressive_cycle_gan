import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import WSConv2d, PixelNorm, ResidualBlock, ConvBlock, factors


class GeneratorCycleGAN(nn.Module):
    def __init__(self, in_channels, img_channels):
        super(GeneratorCycleGAN, self).__init__()
        self.initial = nn.Sequential(
            WSConv2d(img_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm(),
            WSConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm(),
        )

        # Add 6 residual blocks (as in CycleGAN paper)
        self.res_blocks = nn.Sequential(*[ResidualBlock(in_channels) for _ in range(6)])

        self.initial_rgb = WSConv2d(
            in_channels, img_channels, kernel_size=1, stride=1, padding=0
        )
        self.prog_blocks, self.rgb_layers = (
            nn.ModuleList([]),
            nn.ModuleList([self.initial_rgb]),
        )

        for i in range(len(factors) - 1):
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i + 1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c))
            self.rgb_layers.append(
                WSConv2d(conv_out_c, img_channels, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, x, alpha, steps):
        out = self.initial(x)
        out = self.res_blocks(out)  # Pass through residual blocks

        if steps == 0:
            return self.initial_rgb(out)

        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode="nearest")
            out = self.prog_blocks[step](upscaled)

        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out)
        return self.fade_in(alpha, final_upscaled, final_out)

    def fade_in(self, alpha, upscaled, generated):
        # Alpha blend between upscaled and generated images
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)
