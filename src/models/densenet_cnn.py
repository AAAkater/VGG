import torch.nn as nn
from torch import Tensor
from torchvision import models
from torchvision.models import DenseNet121_Weights


class DenseNetNormal(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(DenseNetNormal, self).__init__()
        self.num_classes = num_classes
        self.dense_net = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        for param in self.dense_net.parameters():
            param.requires_grad = False
        self.dense_net.classifier = nn.Linear(
            self.dense_net.classifier.in_features, num_classes
        )

    def forward(self, x: Tensor):
        x = self.dense_net(x)
        return x
