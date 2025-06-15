# 导入必要的库
import time
import torch
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import ImageTxtDataset  # 自定义数据集类


# 定义AlexNet模型类
class AlexNet(torch.nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # 使用Sequential容器按顺序构建网络
        self.model = torch.nn.Sequential(
            # 第一层卷积：输入3通道(RGB)，输出48通道，3x3卷积核，padding=1保持尺寸
            torch.nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1),
            # 最大池化：2x2窗口，步长2（下采样）
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # 第二层卷积：48->128通道
            torch.nn.Conv2d(48, 128, kernel_size=3, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # 第三层卷积：128->192通道
            torch.nn.Conv2d(128, 192, kernel_size=3, padding=1),
            # 第四层卷积：192->192通道
            torch.nn.Conv2d(192, 192, kernel_size=3, padding=1),
            # 第五层卷积：192->128通道
            torch.nn.Conv2d(192, 128, kernel_size=3, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # 展平层：将三维特征图转换为一维向量
            torch.nn.Flatten(),
            # 全连接层：128*4*4 -> 2048
            torch.nn.Linear(128 * 4 * 4, 2048),
            # 全连接层：2048 -> 2048
            torch.nn.Linear(2048, 2048),
            # 输出层：2048 -> 10（假设是10分类任务）
            torch.nn.Linear(2048, 10)
        )

    def forward(self, x):
        # 前向传播：直接按顺序通过Sequential模型
        return self.model(x)


# 数据预处理和加载
# 使用自定义的ImageTxtDataset加载数据
train_data = ImageTxtDataset(
    r"D:\dataset\train.txt",  # 训练集文本文件路径
    r"D:\dataset\image2\train",  # 训练集图片目录
    transforms.Compose([  # 定义数据增强和归一化
        transforms.Resize(224),  # 调整图像大小
        transforms.RandomHorizontalFlip(),  # 随机水平翻转（数据增强）
        transforms.ToTensor(),  # 转换为Tensor格式
        # 归一化（使用ImageNet的均值和标准差）
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
)

# 测试集使用相同的预处理
test_data = ImageTxtDataset(
    r"D:\dataset\train.txt",
    r"D:\dataset\image2\train",
    transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
)

# 创建数据加载器
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)  # 训练集打乱顺序
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)  # 测试集不打乱

# 初始化模型
model = AlexNet()

# 定义损失函数和优化器
loss_fn = torch.nn.CrossEntropyLoss()  # 交叉熵损失（适用于分类任务）
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # SGD优化器

# 训练参数初始化
total_train_step = 0  # 记录训练步数
total_test_step = 0  # 记录测试步数
epoch = 10  # 训练轮数
writer = SummaryWriter("../logs_train")  # TensorBoard日志记录器

start_time = time.time()  # 记录开始时间

# 训练循环
for i in range(epoch):
    print(f"-----第{i+1}轮训练开始-----")
    model.train()  # 设置为训练模式
    for data in train_loader:
        imgs, targets = data  # 获取批次数据和标签
        outputs = model(imgs)  # 前向传播
        loss = loss_fn(outputs, targets)  # 计算损失

        # 反向传播和优化
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新参数

        total_train_step += 1
        # 每500步记录一次训练损失
        if total_train_step % 500 == 0:
            print(f"第{total_train_step}步的训练的loss:{loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 计算并打印本轮训练时间
    end_time = time.time()
    print(f"训练时间{end_time - start_time}")

    # 测试阶段
    model.eval()  # 设置为评估模式
    total_test_loss = 0.0  # 累计测试损失
    total_accuracy = 0  # 累计正确预测数
    with torch.no_grad():  # 禁用梯度计算（节省内存）
        for data in test_loader:
            imgs, targets = data
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()  # 累加损失
            # 计算正确预测的数量（取预测最大概率的类别）
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    # 打印测试结果
    print(f"整体测试集上的loss:{total_test_loss}")
    print(f"整体测试集上的正确率：{total_accuracy / len(test_data)}")
    # 记录测试损失和准确率
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test