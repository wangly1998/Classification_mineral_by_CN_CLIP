import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


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


def load_data(excel_path, sheet_names, image_dirs, train_transform=None,  val_transform=None, test_size=0.2,
                                      valid_size=0.5):
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

    # 分割数据集为训练集、测试集和验证集
    train_data, test_valid_data = train_test_split(combined_data, test_size=test_size, random_state=42)
    test_data, valid_data = train_test_split(test_valid_data, test_size=valid_size, random_state=42)

    # 创建数据集
    datasets = {
        'train': RockDataset(train_data, label_encoder, train_transform),
        'test': RockDataset(test_data, label_encoder, val_transform),
        'valid': RockDataset(valid_data, label_encoder, val_transform)
    }

    return datasets, label_encoder



