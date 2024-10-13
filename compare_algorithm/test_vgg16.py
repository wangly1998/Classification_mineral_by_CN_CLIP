import logging
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from torchvision.models import VGG16_Weights
from sklearn.metrics import precision_score, recall_score, f1_score

class RockDataset(Dataset):
    def __init__(self, data, label_encoder, transform=None):
        self.data = data
        self.label_encoder = label_encoder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_id = str(row['图像编号'])
        label = row['描述']
        image_path = row['图像路径']

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.label_encoder.transform([label])[0]
        label = torch.tensor(label, dtype=torch.long)

        return image, label

def load_data(excel_path, sheet_names, image_dirs, train_transform=None, test_size=0.2):
    # 读取Excel文件中的所有sheets
    all_sheets = pd.read_excel(excel_path, sheet_name=None)  # Read all sheets into a dictionary

    combined_data = pd.DataFrame()

    # 合并所有sheet的数据
    for sheet_name, image_dir in zip(sheet_names, image_dirs):
        data = all_sheets[sheet_name]
        data['图像路径'] = data['图像编号'].apply(lambda x: os.path.join(image_dir, f"{x}.jpg"))
        combined_data = pd.concat([combined_data, data], ignore_index=True)

    # 进行标签编码
    label_encoder = LabelEncoder()
    label_encoder.fit(combined_data['描述'])

    # 分割数据集为训练集和测试集（无验证集）
    train_data, test_data = train_test_split(combined_data, test_size=test_size, random_state=42)

    # 创建数据集
    datasets = {
        'train': RockDataset(train_data, label_encoder, train_transform),
        'test': RockDataset(test_data, label_encoder, val_transform)
    }

    return datasets, label_encoder

# This is the new function for loading validation data only
def _val_load_data(val_excel_path, val_sheet_names, val_image_dirs, val_transform=None):
    # 读取Excel文件中的验证集数据
    all_sheets = pd.read_excel(val_excel_path, sheet_name=None)  # Read all sheets into a dictionary

    combined_data = pd.DataFrame()

    # 合并所有验证集sheet的数据
    for sheet_name, image_dir in zip(val_sheet_names, val_image_dirs):
        data = all_sheets[sheet_name]
        data['图像路径'] = data['图像编号'].apply(lambda x: os.path.join(image_dir, f"{x}.jpg"))
        combined_data = pd.concat([combined_data, data], ignore_index=True)

    # 进行标签编码
    label_encoder = LabelEncoder()
    label_encoder.fit(combined_data['描述'])

    # 创建验证集数据集
    validation_dataset = RockDataset(combined_data, label_encoder, val_transform)

    return validation_dataset, label_encoder

# 设置日志记录
logging.basicConfig(
    filename='training_log_VGG16.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 图像预处理
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC),
    transforms.AutoAugment(transforms.AutoAugmentPolicy('imagenet')),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 验证集和测试集的transform（不使用数据增强）
val_transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 数据路径和配置
excel_path = r"D:\doctor\use_clip_mineral\dataset\scienceDB download\南京大学岩石教学薄片\汇总无描述.xlsx"
sheet_names = ['火成岩', '沉积岩', '变质岩']
image_dirs = [
    r"D:\doctor\use_clip_mineral\dataset\scienceDB download\南京大学岩石教学薄片\南京大学火成岩教学薄片照片数据集",
    r"D:\doctor\use_clip_mineral\dataset\scienceDB download\南京大学岩石教学薄片\南京大学沉积岩教学薄片照片数据集",
    r"D:\doctor\use_clip_mineral\dataset\scienceDB download\南京大学岩石教学薄片\南京大学变质岩教学薄片照片数据集"
]

# 加载数据集
datasets, label_encoder = load_data(excel_path, sheet_names, image_dirs,  train_transform)
train_loader = DataLoader(datasets['train'], batch_size=32, shuffle=True)
test_loader = DataLoader(datasets['test'], batch_size=32, shuffle=False)
valid_loader = DataLoader(datasets['valid'], batch_size=32, shuffle=False)

# 定义设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义模型
model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(label_encoder.classes_))
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# 计算指标的函数
def calculate_metrics(model, data_loader, k_list=[1, 5, 10]):
    model.eval()
    top_k_correct = {k: 0 for k in k_list}
    total_samples = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            _, preds = outputs.topk(max(k_list), dim=1, largest=True, sorted=True)
            total_samples += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds[:, 0].cpu().numpy())  # 只使用 top-1 的预测来计算精确率、召回率和 F1 分数

            for k in k_list:
                correct = preds[:, :k].eq(labels.view(-1, 1).expand_as(preds[:, :k]))
                top_k_correct[k] += correct.sum().item()

    top_k_recall = {k: correct / total_samples for k, correct in top_k_correct.items()}
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    return top_k_recall, precision, recall, f1

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10,save_path='best_model.pt'):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        # 验证阶段
        model.eval()
        with torch.no_grad():
            test_running_corrects = 0
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                test_running_corrects += torch.sum(preds == labels.data)
            test_acc = test_running_corrects.double() / len(test_loader.dataset)

        # 记录当前 epoch 的指标到日志文件
        logging.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Validation Acc: {test_acc:.4f}')

        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, save_path)
    return model
