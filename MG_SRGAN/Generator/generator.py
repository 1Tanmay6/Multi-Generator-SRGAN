
from .upscaling_block import UpscalingBlock
from .residual_block import ResidualBlock
import torch.nn as nn
from ..Utils import get_config_files


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        gen_config = get_config_files(key='generator')
        input_channels = int(gen_config['input_channels'])
        number_of_residuals = int(
            gen_config['number_of_residual_blocks'])
        number_of_channels = int(gen_config['number_of_channels'])
        upscale_factor = int(gen_config['upscale_factor'])
        kernel_size1 = int(gen_config['kernel_size1'])
        kernel_size2 = int(gen_config['kernel_size2'])
        padding1 = int(gen_config['padding1'])
        padding2 = int(gen_config['padding2'])

        self.conv1 = nn.Conv2d(
            input_channels, number_of_channels, kernel_size=kernel_size1, stride=1, padding=padding1)
        self.prelu = nn.PReLU()

        self.residuals = nn.Sequential(
            *[ResidualBlock(number_of_channels) for _ in range(number_of_residuals)])

        self.conv2 = nn.Conv2d(number_of_channels, number_of_channels, kernel_size=kernel_size2,
                               stride=1, padding=padding2, bias=False)
        self.bn2 = nn.BatchNorm2d(number_of_channels)

        self.upscale = nn.Sequential(
            *[UpscalingBlock(number_of_channels, upscale_factor) for _ in range(upscale_factor // 2)])

        self.conv3 = nn.Conv2d(number_of_channels, 3,
                               kernel_size=kernel_size1, stride=1, padding=padding1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu(x)

        residual = x
        x = self.residuals(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual

        x = self.upscale(x)
        x = self.conv3(x)
        return x
