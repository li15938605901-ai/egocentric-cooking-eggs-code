# egocentric-cooking-eggs-code
PyTorch dataloaders and preprocessing scripts for my dataset.

# Egocentric Cooking Eggs Dataset

[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/huamiangeneral/egocentric-cooking-eggs)

这是一个关于第一人称家常烹饪鸡蛋的计算机视觉视频数据集。本仓库提供了针对该数据集的标准 PyTorch 数据加载器（DataLoader），方便研究人员快速上手。

## 📦 1. 下载数据集
我们使用 Hugging Face 托管所有的原始视频文件（.mp4）。请点击上方徽章或访问以下链接下载：
👉 **[Hugging Face 数据集主页](https://huggingface.co/datasets/huamiangeneral/egocentric-cooking-eggs)**

建议您将下载好的 `.mp4` 视频文件放置在本地的 `data/` 目录下。

## 🛠️ 2. 快速使用 (PyTorch)
我们提供了一个轻量且开箱即用的 `dataset.py`。该代码使用 `torchvision.io` 原生读取视频，不依赖复杂的第三方库。

**基础调用示例：**

```python
from dataset import EgocentricEggDataset
from torch.utils.data import DataLoader

# 实例化数据集（请确保 data 文件夹下有您的 mp4 视频）
dataset = EgocentricEggDataset(video_dir='./data')
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 迭代获取视频 Tensor
for videos, video_names in dataloader:
    # videos 的形状默认转换为: [Batch, Frames, Channels, Height, Width]
    print(f"正在处理视频: {video_names}")
    print(f"Tensor 形状: {videos.shape}")
