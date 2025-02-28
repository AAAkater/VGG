import torch.nn as nn
import torchvision.models as models
from torch import Tensor
from torchvision.models import VGG, VGG16_Weights


class VGG16WithCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(VGG16WithCNN, self).__init__()
        self.num_classes = num_classes
        self.vgg16: VGG = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        for param in self.vgg16.parameters():
            param.requires_grad = False

        self.custom_cnn = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 3 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: Tensor):
        x = self.vgg16.features(x)
        x = self.custom_cnn(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG16Normal(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(VGG16Normal, self).__init__()
        self.num_classes = num_classes
        self.vgg16: VGG = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        for param in self.vgg16.parameters():
            param.requires_grad = False

        self.vgg16.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: Tensor):
        x = self.vgg16(x)
        return x
