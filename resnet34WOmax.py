import glob
import os
import numpy as np
import tqdm
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import copy, time
import argparse

from datasets import EarlyStopScheduler, FaceImageDataset, FaceLoadImageDataset
from torchvision import transforms, datasets

from utils import BasicBlock

### basic block for all resnet

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 4000, zero_init_residual = False, norm_layer = None, kernel_size =3):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.in_channels = 64
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=4, stride=1, padding=1, bias=False)

        self.bn1 = norm_layer(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        ## TODO maxpooling or not
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, num_classes)
        self.drop2 = nn.Dropout(p=0.2)
        
        ### Kaiming normalization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, out_channels, blocks, stride= 1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                norm_layer(out_channels))

        layers = []
        layers.append(block(self.in_channels, out_channels, stride = stride, kernel_size = self.kernel_size, 
                            downsample = downsample, norm_layer = norm_layer))
        self.in_channels = out_channels
        
        ## repeat the residule mode
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, kernel_size = self.kernel_size, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        self.embeding  = x
        
        ## try add dropout
        x = self.drop2(x)
        x = self.fc1(x)
        
        return x
    
    def verification(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        self.embeding  = x
        return self.embeding

def test_model(model, test_loader, criterion, device):
    with torch.no_grad():
        model.eval()

        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0

        for batch_idx, (data, target) in enumerate(tqdm.tqdm(test_loader)):   
            data = data.to(device)
            target = target.to(device)
            
            outputs = model(data)

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += target.size(0)
            correct_predictions += (predicted == target).sum().item()
            
            loss = criterion(outputs, target).detach()
            running_loss += loss.item()


        running_loss /= len(test_loader)
        acc = (correct_predictions/total_predictions)*100.0
        print('Testing Loss: ', running_loss)
        print('Testing Accuracy: ', acc, '%')
        return running_loss, acc

def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    model.to(device)
    running_loss, correct, total = 0, 0, 0
    
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(tqdm.tqdm(train_loader)):   
        optimizer.zero_grad()   # .backward() accumulates gradients
        data = data.to(device)
        target = target.to(device) # all data & model on same device
        
        outputs = model(data)
        
        loss = criterion(outputs, target)
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum().item()

        loss.backward()
        optimizer.step()
    
    end_time = time.time()
    
    running_loss /= len(train_loader)
    print('Training Loss: ', running_loss, 'Acc: ', correct/total, 'Time: ',end_time - start_time, 's')
    return running_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hw2 recommend')
    parser.add_argument("--device", type=str, default='cuda:0', help="cuda or cpu")
    parser.add_argument("--save", type=str, default='./default', help="model file to save")
    parser.add_argument("--optm", type=str, default='SGD', help="SGD or Adam")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--min-lr", type=float, default=0.0005, help="minimum lr for ReduceLROnPlateau")
    parser.add_argument("--batch-size", type=int, default=128, help="number of minibatch size")
    parser.add_argument("--milestones", type=int, default=15, help="milestones for applying multiplier")
    parser.add_argument("--epochs", type=int, default=40, help="number of training epochs")
    parser.add_argument("--early-stop", type=int, default=5, help="number of epochs for early stop training")
    parser.add_argument("--patience", type=int, default=5, help="patience for Early Stop")
    parser.add_argument("--factor", type=float, default=0.5, help="ReduceLROnPlateau factor")
    parser.add_argument("--momentum", type=float, default=0, help="momentum of the optimizer")
    parser.add_argument("--seed", type=int, default=0, help='Random seed.')
    args = parser.parse_args(); print(args)
    torch.manual_seed(args.seed)

    num_workers = 8

    train_data_root = "data/train_data"
    val_data_root = "data/val_data"
    test_data_root = "data/test_data"
    
    ## Data augmentation
    Jitter_transforms = transforms.RandomApply([transforms.ColorJitter(brightness = 0.3, contrast = 0.3)], p=0.5)
    flip_transform = transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.5)
    image_transform = transforms.Compose([transforms.ToPILImage(),flip_transform, Jitter_transforms,transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    norm_transform = transforms.Compose([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    valset = FaceLoadImageDataset(val_data_root, "data/val", transform=norm_transform)
    test_loader = DataLoader(valset, batch_size=256, shuffle=False, num_workers=num_workers, drop_last=False)
   
    #from datasets import EarlyStopScheduler
    resnet34 = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=4000)
    model = resnet34
    criterion = nn.CrossEntropyLoss()
    exec('optimizer = torch.optim.%s(model.parameters(), momentum=%f, lr=%f)'%("SGD", 0.1, args.lr))
    scheduler = EarlyStopScheduler(optimizer, factor=args.factor, verbose=True, min_lr=args.min_lr, patience=5)
    device = torch.device(args.device)
    model.to(device)
    print(model)

    Train_loss = []
    Test_loss = []
    Test_acc = []
    best_acc = 0
    
    trainset = FaceLoadImageDataset(train_data_root, "data/train" , valset.target_dict, transform=image_transform)
    train_loader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=num_workers, drop_last=False)
    

    for i in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc = test_model(model, test_loader, criterion)
        Train_loss.append(train_loss)
        Test_loss.append(test_loss)
        Test_acc.append(test_acc)

        with open("./result/result1.txt","a") as file:
            file.write(str(test_acc))

        if test_acc > best_acc:
            print("New best Model, copying...")
            best_acc, best_net = test_acc, copy.deepcopy(model)
            torch.save(best_net, args.save)

        if scheduler.step(error=1-test_acc):
            print('Early Stopping!')
            break





