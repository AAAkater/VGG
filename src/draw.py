import json
import os
from typing import List

from matplotlib import pyplot as plt


def draw(
    train_losses: List[float],
    val_losses: List[float],
    train_accuracies: List[float],
    val_accuracies: List[float],
):
    os.makedirs("plots", exist_ok=True)
    # 绘制损失曲线
    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.grid(True)

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy", color="blue")
    plt.plot(val_accuracies, label="Validation Accuracy", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy Curve")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("../plots/training_curves.png")
    plt.show()


def main():
    file_path = "./training_logs.json"

    with open(file_path, "r") as file:
        data = json.load(file)

    # 提取属性
    train_losses = data["train_losses"]
    val_losses = data["val_losses"]
    train_accuracies = data["train_accuracies"]
    val_accuracies = data["val_accuracies"]

    draw(
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies,
    )


if __name__ == "__main__":
    main()
