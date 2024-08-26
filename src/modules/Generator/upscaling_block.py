import torch.nn as nn


class UpscalingBlock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super(UpscalingBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels *
                              scale_factor ** 2, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.prelu = nn.PReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.pixel_shuffle(out)
        out = self.prelu(out)
        return out
