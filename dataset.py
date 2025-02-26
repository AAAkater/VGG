import torch
from datasets import ClassLabel, load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

data_path = ""

dataset = load_dataset(data_path)


split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
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
    cache_file_name="./cache/cached_test_dataset.arrow",
)
train_dataset = train_dataset.map(
    preprocess,
    batched=True,
    batch_size=100,
    num_proc=1,
    load_from_cache_file=True,
    cache_file_name="./cache/cached_train_dataset.arrow",
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


train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn,
)

if __name__ == "__main__":
    print(len(train_loader))
    print(len(val_loader))
    print(train_dataset.features)
    print(val_dataset.features)
