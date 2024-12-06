import torch
import torch.nn as nn

from modules import WSConv2d, ConvBlock, factors


class DiscriminatorCycleGAN(nn.Module):
    def __init__(self, in_channels, img_channels=3):
        super(DiscriminatorCycleGAN, self).__init__()
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)

        # Mirror the generator's progression
        for i in range(len(factors) - 1, 0, -1):
            conv_in = int(in_channels * factors[i])
            conv_out = int(in_channels * factors[i - 1])
            self.prog_blocks.append(ConvBlock(conv_in, conv_out, use_pixelnorm=False))
            self.rgb_layers.append(
                WSConv2d(img_channels, conv_in, kernel_size=1, stride=1, padding=0)
            )

        # Initial layer for smallest size (4x4)
        self.initial_rgb = WSConv2d(
            img_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.rgb_layers.append(self.initial_rgb)
        self.avg_pool = nn.AvgPool2d(
            kernel_size=2, stride=2
        )  # Down sampling with average pooling

        # Final block for 4x4 image input
        self.final_block = nn.Sequential(
            WSConv2d(in_channels + 1, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=4, padding=0, stride=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, 1, kernel_size=1, padding=0, stride=1),
        )

    def fade_in(self, alpha, downscaled, out):
        """Blends downscaled and processed outputs during progression."""
        return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(self, x):
        """Adds a minibatch standard deviation channel to the input."""
        batch_statistics = (
            (torch.std(x, dim=0, unbiased=False) + 1e-8)
            .mean()
            .repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        )
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x, alpha, steps):
        """
        Forward pass through the discriminator.
        x: Input image tensor.
        alpha: Fade-in alpha value.
        steps: Number of progressive steps.
        """
        cur_step = len(self.prog_blocks) - steps

        # Initial conversion from RGB
        out = self.leaky(self.rgb_layers[cur_step](x))

        if steps == 0:  # If image size is 4x4
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)

        # Downscale input and process through progressive blocks
        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[cur_step](out))
        out = self.fade_in(alpha, downscaled, out)

        # Continue through remaining blocks
        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        # Final processing
        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)
