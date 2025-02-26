import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# 加载预训练的 VGG16 模型
model = models.vgg16(pretrained=False)  # 不使用预训练权重
num_classes = 3  # 根据你的类别数量调整
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)


model_path = "./vgg_model/vgg16_cnn_model_2_1.pth"
# 加载训练好的权重
model.load_state_dict(torch.load(model_path))
model.eval()  # 设置为评估模式


# 定义图像预处理
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # 归一化
    ]
)

# 加载测试数据集
test_dataset = datasets.ImageFolder(root="path_to_test_data", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

correct = 0
total = 0
with torch.no_grad():  # 禁用梯度计算
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")


all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 生成混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)

# 生成分类报告
class_names = test_dataset.classes
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))
