import torch.nn as nn
from torch import Tensor
from torchvision import models
from torchvision.models import DenseNet121_Weights


class DenseNetNormal(nn.Module):
    def __init__(self, num_classes=10):
        super(DenseNetNormal, self).__init__()
        self.dense_net = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        in_features = self.dense_net.classifier.in_features
        self.dense_net.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: Tensor):
        return self.dense_net(x)
