import torch.nn as nn
from torch import Tensor
from torchvision import models
from torchvision.models import ResNet50_Weights


class ResNetNormal(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(ResNetNormal, self).__init__()
        self.num_classes = num_classes
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        for param in self.resnet.parameters():
            param.requires_grad = False

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x: Tensor):
        x = self.resnet(x)
        return x
