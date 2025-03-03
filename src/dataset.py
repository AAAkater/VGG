import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import ClassLabel, load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

data_path = "/home/ym/code/python/datasets/Rice-Leaf-Disease"

dataset = load_dataset(data_path)


split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        # transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def preprocess(examples):
    examples["image"] = [transform(image.convert("RGB")) for image in examples["image"]]
    return examples


val_dataset = val_dataset.map(
    preprocess,
    batched=True,
    batch_size=100,
    num_proc=1,
    load_from_cache_file=True,
    cache_file_name="../.cache/cached_test_dataset.arrow",
    # cache_file_name="../.cache/cached_test_dataset_inceptionv3.arrow",
)
train_dataset = train_dataset.map(
    preprocess,
    batched=True,
    batch_size=100,
    num_proc=1,
    load_from_cache_file=True,
    cache_file_name="../.cache/cached_train_dataset.arrow",
    # cache_file_name="../.cache/cached_train_dataset_inceptionv3.arrow",
)

train_dataset.set_format("torch", columns=["image", "label"])
val_dataset.set_format("torch", columns=["image", "label"])


def collate_fn(batch):
    images = [item["image"] for item in batch]
    labels = [item["label"] for item in batch]

    # 将 images 和 labels 转换为张量
    images = torch.stack(images)
    labels = torch.tensor(labels)  # 将 labels 转换为张量
    return images, labels


small_batch_size = 32

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=small_batch_size,
    shuffle=True,
    collate_fn=collate_fn,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=small_batch_size,
    shuffle=False,
    collate_fn=collate_fn,
)


def draw():
    # 获取训练集和测试集的标签
    train_labels = train_dataset["label"]
    val_labels = val_dataset["label"]

    # 计算训练集和测试集中各个标签的数量
    train_label_counts = np.bincount(train_labels)
    val_label_counts = np.bincount(val_labels)

    # 获取标签的名称（假设你已经知道标签的名称）
    label_names = [
        "Bacterialblight",
        "Blast",
        "Brownspot",
        "Healthy",
        "Tungro",
    ]  # 替换为实际的标签名称

    # 创建一个 DataFrame 来存储数据
    data = pd.DataFrame(
        {
            "Label": label_names,
            "Train": train_label_counts,
            "Validation": val_label_counts,
        }
    )

    # 将数据转换为长格式，方便 seaborn 绘图
    data_melted = data.melt(id_vars="Label", var_name="Dataset", value_name="Count")

    # 绘制分组柱状图
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Label", y="Count", hue="Dataset", data=data_melted, palette="Set2")
    plt.title("Label Distribution in Train and Validation Datasets")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.legend(title="Dataset")
    plt.tight_layout()
    plt.savefig("../plots/dataset_chart.png")
    plt.show()


if __name__ == "__main__":
    print(len(train_loader))
    print(len(val_loader))
    print(train_dataset.features)
    print(val_dataset.features)

    draw()
