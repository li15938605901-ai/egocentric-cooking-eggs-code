在深度学习训练中，我们需要把 `.mp4` 文件变成模型能看懂的张量（Tensor）。这里为你提供一个基于 PyTorch 和 `torchvision` 的标准写法：

```python
import os
import torch
from torch.utils.data import Dataset
import torchvision.io as io

class EgocentricEggDataset(Dataset):
    def __init__(self, video_dir, transform=None):
        """
        初始化数据集
        :param video_dir: 存放 mp4 视频的文件夹路径
        :param transform: 视频增强或预处理操作 (如 Resize, Normalize)
        """
        self.video_dir = video_dir
        self.transform = transform
        # 获取目录下所有的 mp4 文件列表
        self.video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

    def __len__(self):
        # 返回数据集中视频的总数
        return len(self.video_files)

    def __getitem__(self, idx):
        # 拼接出单个视频的完整路径
        video_path = os.path.join(self.video_dir, self.video_files[idx])
        
        # 使用 torchvision 读取视频
        # vframes 是形状为 (T, H, W, C) 的 Tensor，其中 T 是帧数
        vframes, aframes, info = io.read_video(video_path, pts_unit='sec')
        
        # PyTorch 习惯的维度是 (C, T, H, W) 或图像的 (C, H, W)，这里进行转换
        vframes = vframes.permute(3, 0, 1, 2)
        
        # 归一化到 0-1 之间，并将类型转为 float32
        vframes = vframes.to(torch.float32) / 255.0

        if self.transform:
            vframes = self.transform(vframes)
            
        # 这里为了演示，暂时返回一个假的 label = 0
        label = 0 
        
        return vframes, label

# 测试代码是否能跑通
if __name__ == '__main__':
    # 假设你当前目录下有个 data 文件夹，里面有视频
    # dataset = EgocentricEggDataset(video_dir='./data')
    # print(f"加载了 {len(dataset)} 个视频。")
    pass
