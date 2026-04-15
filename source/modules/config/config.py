from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TestConfig:
    test_suffix: str = 'PNG'   # 测试数据的图像文件后缀
    test_maxNumberOfImages: int = 10  # 测试时每个物体的图像数量
    test_datatype: str = 'RealData'  # 测试数据集类型
    test_prefix: str = '0*.png'  # 测试数据前缀
    image_size: int = 512  # 测试时输入图像的分辨率 (高=宽); 输入图像在编码前会被缩放到此尺寸

@dataclass
class TrainConfig:
    train_suffix: str = 'data'  # 训练数据的图像文件后缀
    train_maxNumberOfImages: int = 10  # 训练时每个物体的图像数量
    image_size: int = 512  # 训练时输入图像的分辨率 (高=宽); 输入图像在编码前会被缩放到此尺寸
    train_datatype: str = 'AdobeNPI'  # 训练数据集类型
    train_prefix: str = '0*.tif'  # 训练数据前缀

@dataclass
class Config:
    train: TrainConfig = field(default_factory=TrainConfig)
    test: TestConfig = field(default_factory=TestConfig)

config = Config()
