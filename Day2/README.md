# DAY2 深度学习实验笔记：CNN在CIFAR-10上的实现与优化

## 一、实验概览
**核心目标**：实现两种CNN模型（AlexNet改进版/Custom-ChenNet）在CIFAR-10数据集的分类任务  
**技术栈**：PyTorch + TensorBoard + GPU加速  
**关键收获**：
- 掌握CNN架构设计中的尺寸匹配原则
- 理解完整模型训练流程的标准化实现
- 学会处理实际训练中的过拟合问题

## 二、模型架构对比
### 1. AlexNet改进版

### 主要修改点（适配32x32输入）
nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 原版11x11→3x3
    nn.MaxPool2d(2, 2),                                  # 池化窗口减小
    ...
    nn.Linear(256*4*4, 4096)                             # 全连接层输入维度调整
)


### 结构特点

- **5层卷积 + 3层池化 → 全连接层**
- 使用**自适应池化**统一输出尺寸
- 添加**Dropout层**（p=0.5）防止过拟合

### 2. Custom-ChenNet
nn.Sequential(
    nn.Conv2d(3, 32, 5, padding=2),
    nn.MaxPool2d(2),
    ... # 3层卷积-池化组合
    nn.Linear(1024, 10)  # 直接输出10分类
)
- 优势：参数量更小，训练速度更快

## 三、关键技术要点

## 1. 输入尺寸适配
- **问题**：原始AlexNet设计用于224x224输入，直接应用于32x32的CIFAR-10会导致特征图尺寸归零。
- **解决方案**：
  - 减小卷积核尺寸（11→3）
  - 调整步长（4→1）
  - 修改池化窗口（3→2）

## 2. 过拟合处理
- **数据层面**：后续可添加`RandomHorizontalFlip()`等增强。
- **模型层面**：
  ```python
  nn.Dropout(0.5)  # 随机失活
  nn.BatchNorm2d() # 可选项

## 3. 模型部署技巧
#保存与加载最佳模型
torch.save(model.state_dict(), "best.pth")
model.load_state_dict(torch.load("best.pth"))
#推理模式设置
model.eval()
with torch.no_grad():
    outputs = model(inputs)
