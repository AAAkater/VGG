import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights

vgg16 = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

for param in vgg16.parameters():
    param.requires_grad = False


num_classes = 5
vgg16.classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096),  # VGG16 的原始全连接层
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(4096, num_classes),  # 输出类别数
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg16 = vgg16.to(device)
