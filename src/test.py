import torch

from dataset import val_loader
from models.googlenet_cnn import GoogleNetNormal

num_classes = 5
model = GoogleNetNormal()  # 你的模型
model.eval()  # 设置模型为评估模式

# 如果有 GPU，将模型移动到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 禁用梯度计算
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = model(inputs)

        # 获取预测结果
        _, predicted = torch.max(outputs.data, 1)

        # 统计正确预测的数量
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# 打印测试集上的准确率
print(f"Accuracy on test set: {100 * correct / total:.2f}%")
