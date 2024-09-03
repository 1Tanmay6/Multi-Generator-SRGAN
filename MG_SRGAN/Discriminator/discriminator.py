import torch.nn as nn
from ..Utils import get_config_files


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        dis_config = get_config_files(key='discriminator')
        input_channels = int(dis_config['input_channels'])
        input_size = int(dis_config['input_size'])
        fc_size = int(dis_config['fc_size'])
        leaky_relu_slope = float(dis_config['leaky_relu_slope'])

        def discriminator_block(in_filters, out_filters, stride):
            return nn.Sequential(
                nn.Conv2d(in_filters, out_filters, kernel_size=3,
                          stride=stride, padding=1),
                nn.BatchNorm2d(out_filters),
                nn.LeakyReLU(leaky_relu_slope, inplace=True)
            )

        self.model = nn.Sequential(
            *discriminator_block(input_channels, 32, stride=1),
            *discriminator_block(32, 64, stride=4),
            *discriminator_block(64, 128, stride=4),
        )

        # Calculate the size after convolutions
        conv_output_size = input_size // 16  # Downscaled twice by stride=4

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(128 * conv_output_size * conv_output_size, fc_size),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Linear(fc_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
