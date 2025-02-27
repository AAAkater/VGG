# import torch
# import torchvision.models as models
# from PIL import Image
# from torchvision import transforms

# # 定义模型架构
# model = models.vgg16(weights=None)  # 不使用预训练权重
# num_classes = 5  # 假设有 5 个类别
# model.classifier[6] = torch.nn.Linear(4096, num_classes)  # 修改分类器

# model_path = "./vgg_model/vgg16_cnn_model_2.pth"

# model.load_state_dict(torch.load(model_path))


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# model.eval()


# # 定义图像预处理
# transform = transforms.Compose(
#     [
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ]
# )


# if __name__ == "__main__":
#     img_path = ""
#     # 加载图像
#     image = Image.open(img_path).convert("RGB")  # 打开图像并转换为 RGB
#     image = transform(image).unsqueeze(0)  # 添加 batch 维度
#     image = image.to(device)
#     # 推理
#     with torch.no_grad():
#         output = model(image)
#         _, predicted = torch.max(output, 1)
#         print(f"Predicted class: {predicted.item()}")
import torch
import torch.nn as nn

# 定义一个简单的输入张量，模拟一个包含2个样本的图像数据
# 每个样本有3个通道，高度为4，宽度为4
input_tensor = torch.randn(2, 3, 4, 4)
print("输入张量的形状:", input_tensor.shape)
print(input_tensor)
# 创建一个Flatten层
flatten_layer = nn.Flatten()

# 对输入张量进行扁平化处理
output_tensor: torch.Tensor = flatten_layer(input_tensor)
print("输出张量的形状:", output_tensor.shape)
print(output_tensor)
