import torch.nn as nn


class GetLoss:
    def get_adversarial_loss(self):
        return nn.BCELoss()

    def get_content_loss(self):
        return nn.MSELoss()
