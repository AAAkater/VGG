import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models
from torchvision.models import Inception_V3_Weights


class InceptionV3Normal(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(InceptionV3Normal, self).__init__()
        self.num_classes = num_classes
        self.inception_v3 = models.inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1
        )

        for param in self.inception_v3.parameters():
            param.requires_grad = False

        self.inception_v3.fc = nn.Linear(self.inception_v3.fc.in_features, num_classes)

    def forward(self, x: Tensor):
        x = self.inception_v3(x)
        return x
