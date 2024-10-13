import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

from rock_dataset import load_data, DataLoader
from torchvision.models import Inception_V3_Weights
from sklearn.metrics import precision_score, recall_score, f1_score
import logging

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


# 训练模型的函数
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # 处理 Inception 模型的辅助分类器输出
            if model._get_name() == 'Inception3' and model.training:
                outputs, aux_outputs = model(inputs)
                loss1 = criterion(outputs, labels)
                loss2 = criterion(aux_outputs, labels)
                loss = loss1 + 0.4 * loss2
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
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
        logging.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, test Acc: {test_acc:.4f}')

        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = model.state_dict()

    model.load_state_dict(best_model_wts)
    return model

# 设置日志记录
logging.basicConfig(
    filename='training_log_inceptionv3.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 训练集的数据增强
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(299, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC),
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
datasets, label_encoder = load_data(excel_path, sheet_names, image_dirs,  train_transform, val_transform)
train_loader = DataLoader(datasets['train'], batch_size=32, shuffle=True)
valid_loader = DataLoader(datasets['valid'], batch_size=32, shuffle=False)
test_loader = DataLoader(datasets['test'], batch_size=32, shuffle=False)

# 定义设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义模型
model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(label_encoder.classes_))  # 调整分类层
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 训练和评估
trained_model = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=20)

# 计算测试集上的指标
top_k_recall, precision, recall, f1 = calculate_metrics(trained_model, valid_loader, k_list=[1, 5, 10])

# 打印并记录结果
for k, recall_k in top_k_recall.items():
    logging.info(f'Top-{k} Recall: {recall_k:.4f}')
logging.info(f'Precision: {precision:.4f}')
logging.info(f'Recall: {recall:.4f}')
logging.info(f'F1 Score: {f1:.4f}')

