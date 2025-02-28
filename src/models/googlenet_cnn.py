import torch
from torch import Tensor, nn
from torchvision.models import GoogLeNet_Weights, googlenet


class GoogleNetWithCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(GoogleNetWithCNN, self).__init__()
        self.num_classes = num_classes
        self.googlenet = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)

        for param in self.googlenet.parameters():
            param.requires_grad = False

        self.custom_cnn = nn.Sequential(
            nn.Conv2d(1024, 64, kernel_size=3, stride=1, padding=1),
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
        x = self.googlenet(x)
        x = self.custom_cnn(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class GoogleNetNormal(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(GoogleNetNormal, self).__init__()
        self.num_classes = num_classes
        self.googlenet = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)

        for param in self.googlenet.parameters():
            param.requires_grad = False

        self.googlenet.fc = nn.Linear(self.googlenet.fc.in_features, num_classes)

    def forward(self, x: Tensor):
        x = self.googlenet(x)
        return x
