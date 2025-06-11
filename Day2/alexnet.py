import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        """AlexNet模型初始化
        Args:
            num_classes (int): 输出类别数，默认1000(ImageNet类别数)
        """
        super(AlexNet, self).__init__()

        # 特征提取部分(卷积层+池化层)
        self.features = nn.Sequential(
            # 修改第1层：减小卷积核和步长 (原11x11, stride=4 → 3x3, stride=1)
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 32x32 -> 32x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16

            # 修改第2层：保持5x5卷积但减小池化窗口
            nn.Conv2d(64, 192, kernel_size=5, padding=2),  # 16x16 -> 16x16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8

            # 第3层：保持3x3卷积
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # 8x8 -> 8x8
            nn.ReLU(inplace=True),

            # 第4层：保持3x3卷积
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # 8x8 -> 8x8
            nn.ReLU(inplace=True),

            # 第5层：保持3x3卷积
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 8x8 -> 8x8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8 -> 4x4 (最终特征图尺寸)
        )

        # 自适应平均池化层，将特征图统一调整为6x6大小
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # 分类器部分(全连接层)
        self.classifier = nn.Sequential(
            nn.Dropout(),  # 随机失活50%神经元(原论文设定)
            # 全连接层: 256*6*6 -> 4096
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(),  # 第二个dropout层
            # 全连接层: 4096 -> 4096
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            # 输出层: 4096 -> num_classes
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        """前向传播过程
        Args:
            x (torch.Tensor): 输入张量，形状应为[batch_size, 3, 224, 224]
        Returns:
            torch.Tensor: 输出分类结果，形状为[batch_size, num_classes]
        """
        # 特征提取
        x = self.features(x)  # 通过卷积层提取特征

        # 自适应池化
        x = self.avgpool(x)  # 将特征图统一调整为6x6

        # 展平特征图: [batch, 256, 6, 6] -> [batch, 256*6*6]
        x = torch.flatten(x, 1)

        # 分类器处理
        x = self.classifier(x)  # 通过全连接层得到分类结果

        return x


# 测试代码
if __name__ == "__main__":
    # 创建AlexNet实例(默认1000类)
    model = AlexNet()

    # 生成模拟输入数据: batch_size=64, 3通道, 224x224图像
    # 对应形状: [64, 3, 224, 224]
    input_tensor = torch.randn(64, 3, 224, 224)

    # 前向传播
    output = model(input_tensor)

    # 打印输入输出形状
    print(f"输入张量形状: {input_tensor.shape}")  # 应为 torch.Size([64, 3, 224, 224])
    print(f"输出张量形状: {output.shape}")  # 应为 torch.Size([64, 1000])