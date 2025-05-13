import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from torch.utils.data import random_split
import os
import time
import copy

# Set data directory
data_dir = 'EX2/flower_dataset'

# Data augmentation and normalization for training and validation
data_transforms = transforms.Compose([
        # GRADED FUNCTION: Add five data augmentation methods, Normalizating and Tranform to tensor
        ### START SOLUTION HERE ###
        # Add five data augmentation methods, Normalizating and Tranform to tensor
    transforms.RandomRotation(30),       # 随机旋转 (±30度)
    transforms.RandomHorizontalFlip(),   # 水平翻转 (50%概率)
    transforms.RandomVerticalFlip(),     # 垂直翻转 (50%概率)
    transforms.ColorJitter(              # 颜色抖动
        brightness=0.2, contrast=0.2, 
        saturation=0.2, hue=0.1
    ),
    transforms.RandomResizedCrop(224),   # 随机裁剪并缩放到224x224
    transforms.ToTensor(),               # 转换为Tensor
    transforms.Normalize(                # 标准化 (ImageNet参数)
        [0.485, 0.456, 0.406], 
        [0.229, 0.224, 0.225]
    )
        ### END SOLUTION HERE ###
])

# Load the entire dataset
full_dataset = datasets.ImageFolder(data_dir, data_transforms)

# Automatically split into 80% train and 20% validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Use DataLoader for both train and validation datasets
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

dataloaders = {'train': train_loader, 'val': val_loader}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

# Get class names from the dataset
class_names = full_dataset.classes

# Load pre-trained model and modify the last layer
model = models.resnet18(pretrained=True)


# GRADED FUNCTION: Modify the last fully connected layer of model
### START SOLUTION HERE ###
num_classes = len(class_names)           # 获取花朵类别总数
model.fc = nn.Linear(model.fc.in_features, num_classes)
# Modify the last fully connected layer of model
### END SOLUTION HERE ###



# GRADED FUNCTION: Define the loss function
### START SOLUTION HERE ###
# Define the loss function
criterion = nn.CrossEntropyLoss()    
### END SOLUTION HERE ###

# GRADED FUNCTION: Define the optimizer
### START SOLUTION HERE ###
# Define the optimizer
optimizer = optim.SGD(
    model.parameters(), 
    lr=0.001,          # 初始学习率
    momentum=0.9       # 动量加速收敛
)
### END SOLUTION HERE ###

# Learning rate scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training function
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Print learning rate for current epoch
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Learning Rate: {current_lr:.6f}')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        
                        # GRADED FUNCTION: Backward pass and optimization
                        ### START SOLUTION HERE ###
                        loss.backward()         # 反向传播计算梯度
                        optimizer.step()        # 根据梯度更新权重
                        # Backward pass and optimization
                        ### END SOLUTION HERE ###

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()  # Update learning rate based on scheduler

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Save the model if validation accuracy is the best so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # Save the best model
                save_dir = 'Ex2/work_dir'
                os.makedirs(save_dir, exist_ok=True)

               # GRADED FUNCTION: Save the best model
                ### START SOLUTION HERE ###
                # Save the best model
                torch.save(
                    model.state_dict(),             # 保存模型参数
                    os.path.join(save_dir, 'best_model.pth')  # 保存路径
                )
                ### END SOLUTION HERE ###

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model

# Train the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model = train_model(model, criterion, optimizer, scheduler, num_epochs=25)
