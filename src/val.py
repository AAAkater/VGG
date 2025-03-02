import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from dataset import val_loader
from models.vgg_cnn import VGG16WithCNN


def getModel(device: torch.device):
    model = VGG16WithCNN(5)

    model_path = "../train_model/vgg16_cnn_model_40_优化rate.pth"
    # 加载训练好的权重
    model.load_state_dict(
        torch.load(
            model_path,
            weights_only=True,
        )
    )
    model.eval()  # 设置为评估模式
    model.to(device)
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = getModel(device)
    all_labels = []
    all_preds = []

    # 进行推理
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    # 生成混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_preds)
    # 打印混淆矩阵
    print("Confusion Matrix:")
    print(conf_matrix)
    class_names = [
        "Bacterialblight",
        "Blast",
        "Brownspot",
        "Healthy",
        "Tungro",
    ]
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    # 保存混淆矩阵图片
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()
