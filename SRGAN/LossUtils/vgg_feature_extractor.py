import torchvision
import torch.nn as nn

from ..Utils import get_config_files


class VGGFeatureExtractor(nn.Module):
    def __init__(self, use_cuda=True):
        super(VGGFeatureExtractor, self).__init__()
        vgg_config = get_config_files(key='vgg')
        feature_layer = int(vgg_config['feature_layer'])
        vgg19 = torchvision.models.vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(
            *list(vgg19.features.children())[:feature_layer])
        if use_cuda:
            self.feature_extractor = self.feature_extractor.cuda()

    def forward(self, img):
        return self.feature_extractor(img)
