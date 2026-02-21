import os
import torch
from torch.utils.data import Dataset
import torchvision.io as io

class EgocentricEggDataset(Dataset):
    def __init__(self, video_dir, transform=None):
        """
        第一人称煎蛋数据集加载器
        :param video_dir: 存放 mp4 视频的文件夹路径
        :param transform: 视频增强或预处理操作 (如 torchvision.transforms)
        """
        self.video_dir = video_dir
        self.transform = transform
        
        # 遍历目录，获取所有 mp4 文件的列表
        self.video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        
        if len(self.video_files) == 0:
            print(f"警告：在 {video_dir} 目录下没有找到 .mp4 文件！")

    def __len__(self):
        # 返回数据集中视频的总数
        return len(self.video_files)

    def __getitem__(self, idx):
        # 获取当前视频的文件名和完整路径
        video_name = self.video_files[idx]
        video_path = os.path.join(self.video_dir, video_name)
        
        # 使用 torchvision 读取视频
        # vframes 的原始形状为 (T, H, W, C)，其中 T 是总帧数
        vframes, aframes, info = io.read_video(video_path, pts_unit='sec')
        
        # 将维度转换为 PyTorch 在视频任务中常用的 (T, C, H, W)
        vframes = vframes.permute(0, 3, 1, 2)
        
        # 归一化到 0.0 - 1.0 之间，并将数据类型转为 float32
        vframes = vframes.to(torch.float32) / 255.0

        # 如果定义了数据增强（如 Resize, RandomCrop 等），在这里应用
        if self.transform:
            vframes = self.transform(vframes)
            
        # 返回视频的 Tensor 数据和对应的文件名，方便后续任务追踪
        return vframes, video_name

# ==========================================
# 本地测试代码
# ==========================================
if __name__ == '__main__':
    # 测试前请确保当前目录下有一个名为 'data' 的文件夹，里面放几个 mp4 视频
    test_dir = './data'
    
    if os.path.exists(test_dir):
        dataset = EgocentricEggDataset(video_dir=test_dir)
        print(f"成功加载数据集，共包含 {len(dataset)} 个视频。")
        
        if len(dataset) > 0:
            sample_video, sample_name = dataset[0]
            print(f"测试读取第一个视频：{sample_name}")
            print(f"视频 Tensor 形状 (Frames, Channels, Height, Width): {sample_video.shape}")
            print(f"数据类型: {sample_video.dtype}, 最大值: {sample_video.max():.2f}, 最小值: {sample_video.min():.2f}")
    else:
        print(f"未找到测试目录 {test_dir}，请先创建文件夹并放入视频。")
