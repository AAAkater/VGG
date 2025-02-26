import os
from typing import List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from dataset import train_loader, val_loader
from model import device, vgg16

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(vgg16.classifier.parameters(), lr=0.0001, weight_decay=1e-5)

scheduler = StepLR(optimizer, step_size=2, gamma=0.1)


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计损失和准确率
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(dataloader)
    train_acc = 100.0 * correct / total
    return train_loss, train_acc


# 验证函数
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / len(dataloader)
    val_acc = 100.0 * correct / total
    return val_loss, val_acc


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
    plt.savefig("plots/training_curves.png")
    plt.show()


if __name__ == "__main__":
    num_epochs = 5
    train_losses = []  # 记录训练损失
    val_losses = []  # 记录验证损失
    train_accuracies = []  # 记录训练准确率
    val_accuracies = []  # 记录验证准确率
    for epoch in range(num_epochs):
        print(f"starting epoch {epoch + 1}")
        train_loss, train_acc = train(
            vgg16,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        val_loss, val_acc = validate(
            vgg16,
            val_loader,
            criterion,
            device,
        )
        scheduler.step()
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        if (epoch + 1) % 5 == 0:
            torch.save(vgg16.state_dict(), f"./dd/vgg16_cnn_model_{epoch + 1}.pth")

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Train Loss: {train_loss:.8f}, Train Acc: {train_acc:.8f}%, "
            f"Val Loss: {val_loss:.8f}, Val Acc: {val_acc:.8f}%"
        )
    draw(
        train_losses=train_losses,
        val_losses=val_losses,
        train_accuracies=train_accuracies,
        val_accuracies=val_accuracies,
    )
    print("done")
