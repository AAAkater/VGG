# VGG

## 1. 开始

下载依赖

```bash
cd VGG
pip install -r requirements.txt
```

## 2. 数据集下载并处理

先下数据集(需要科学上网)

```bash
git lfs install
git clone https://huggingface.co/datasets/sharmin3/Rice-Leaf-Disease
```

处理数据集

```bash
python dataset.py
```
