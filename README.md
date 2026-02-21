# egocentric-cooking-eggs-code
PyTorch dataloaders and preprocessing scripts for my dataset.

# Egocentric Cooking Eggs Dataset

[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/huamiangeneral/egocentric-cooking-eggs)

这是一个关于第一人称家常烹饪鸡蛋的计算机视觉视频数据集。本仓库包含该数据集的 PyTorch 加载器和预处理脚本。

## 📦 1. 下载数据集
我们使用 Hugging Face 托管视频文件。请点击上方徽章或访问以下链接下载：
👉 **[Hugging Face 下载地址](https://huggingface.co/datasets/huamiangeneral/egocentric-cooking-eggs)**

您可以将下载的 `.mp4` 视频文件放置在本地的 `data/` 目录下。

## 🛠️ 2. 快速使用 (PyTorch)
我们提供了一个开箱即用的 `dataset.py`。
```python
from dataset import EgocentricEggDataset
from torch.utils.data import DataLoader

# 实例化数据集
dataset = EgocentricEggDataset(video_dir='./data')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 迭代获取视频 Tensor
for videos, labels in dataloader:
    print(videos.shape) # 输出例如: torch.Size([4, 3, 300, 224, 224])
