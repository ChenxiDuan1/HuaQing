# Day3 学习笔记：PyTorch数据集处理与准备

## 主要内容

今天主要学习了如何使用PyTorch创建自定义数据集类以及如何准备和划分图像数据集。以下是关键代码和概念的总结：

### 1. 自定义数据集类 ImageTxtDataset

```python
import os
from torch.utils import data

class ImageTxtDataset(data.Dataset):
    def __init__(self, txt_path: str, folder_name, transform):
        self.transform = transform
        self.data_dir = os.path.dirname(txt_path)
        self.imgs_path = []
        self.labels = []
        self.folder_name = folder_name
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            img_path, label = line.split()
            label = int(label.strip())
            self.labels.append(label)
            self.imgs_path.append(img_path)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, i):
        path, label = self.imgs_path[i], self.labels[i]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label
```
### 收获：

-学会了如何继承torch.utils.data.Dataset类创建自定义数据集  
-理解了必须实现的两个核心方法：__len__和__getitem__  
-掌握了从文本文件读取图像路径和标签的方法  
-了解了数据转换(transform)在数据集类中的应用  

### 2. 数据集划分脚本
```python
import os
import shutil
from sklearn.model_selection import train_test_split
import random

# 设置随机种子以确保可重复性
random.seed(42)

# 数据集路径和划分比例
dataset_dir = 'path/to/dataset'
train_dir = 'path/to/train'
val_dir = 'path/to/val'
train_ratio = 0.7

# 创建目录和划分数据集
for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)
    images = [os.path.join(class_name, img) for img in os.listdir(class_path)]
    train_images, val_images = train_test_split(images, train_size=train_ratio, random_state=42)
    
    # 复制文件到相应目录
    for img in train_images:
        shutil.move(os.path.join(dataset_dir, img), os.path.join(train_dir, img))
    for img in val_images:
        shutil.move(os.path.join(dataset_dir, img), os.path.join(val_dir, img))
```
### 收获：
-学会了使用sklearn.model_selection.train_test_split进行数据集划分  
-掌握了保持类别平衡的数据集划分方法  
-了解了设置随机种子(random seed)的重要性  
-实践了使用shutil模块进行文件操作  

### 3. 数据集准备脚本(prepare.py)
```python
import os

def create_txt_file(root_dir, txt_filename):
    with open(txt_filename, 'w') as f:
        for label, category in enumerate(os.listdir(root_dir)):
            category_path = os.path.join(root_dir, category)
            if os.path.isdir(category_path):
                for img_name in os.listdir(category_path):
                    img_path = os.path.join(category_path, img_name)
                    f.write(f"{img_path} {label}\n")
```
### 收获：
-学会了自动生成数据集描述文件的方法  
-掌握了将图像路径和标签写入文本文件的技巧  
-理解了如何为不同类别分配数字标签  

## 总结
今天的学习让我完整掌握了PyTorch数据集处理的流程：  
原始数据组织 → 2. 数据集划分 → 3. 生成描述文件 → 4. 创建自定义Dataset类  

### 关键收获：  
（1）理解了数据集类在PyTorch训练流程中的作用  
（2）学会了保持数据可重复性的方法(随机种子)  
（3）掌握了处理图像分类数据集的标准流程  
（4）了解了如何将数据集准备过程自动化  
（5）这些技能对于后续构建完整的深度学习训练流程至关重要，为明天的模型训练部分打下了坚实基础。  
