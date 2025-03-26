from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import argparse
import csv
import time

from models import *
from utils import progress_bar
from randomaug import RandAugment
from models.vit import ViT
from models.convmixer import ConvMixer
from models.mobilevit import mobilevit_xxs

print("可见的GPU列表:", torch.cuda.device_count())
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
torch.cuda.empty_cache()

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')  # resnets: 1e-3, Vit: 1e-4
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--noaug', action='store_false', help='disable use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--nowandb', action='store_true', help='disable wandb')
parser.add_argument('--mixup', action='store_true', help='add mixup augmentations')
parser.add_argument('--net', default='vit')
parser.add_argument('--dp', action='store_true', help='use data parallel')
parser.add_argument('--bs', default='512')
parser.add_argument('--size', default="32")
parser.add_argument('--n_epochs', type=int, default='200')
parser.add_argument('--patch', default='16', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="512", type=int)
parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")
args = parser.parse_args()

# wandb 配置（如果使用 wandb）
# usewandb = ~args.nowandb
usewandb = False
if usewandb:
    import wandb
    watermark = "{}_lr{}".format(args.net, args.lr)
    wandb.init(project="FashionMNIST-challange", name=watermark)
    wandb.config.update(args)

bs = int(args.bs)
imsize = int(args.size)
use_amp = not args.noamp
aug = False

# 指定主设备为 GPU 5（不做映射，直接使用原始设备编号）
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
best_acc = 0   # best test accuracy
start_epoch = 0   # start from epoch 0 or last checkpoint epoch

# Data preparation
print('==> Preparing data..')
if args.net == "vit_timm":
    size = 384
else:
    size = imsize

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, root_dir,is_test=False, transform=None):
        """
        Args:
            csv_file (string): CSV文件的路径，其中第一列是图片文件名，第二列是标签。
            root_dir (string): 图片存放的文件夹路径。
            transform (callable, optional): 施加于图片上的变换。
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test

        if is_test:
            self.data['label'] = -1 

        # 获取类别映射，例如：{'cat': 0, 'dog': 1, ...}
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(self.data['label'].unique()))}


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 组合图片完整路径
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        # 使用PIL打开图片，并确保转换为RGB（3通道）
        image = Image.open(img_name).convert('RGB')
        label = self.label_to_idx[self.data.iloc[idx, 1]]
        if self.transform:
            image = self.transform(image)
        return image, label

print('==> Preparing custom dataset..')

train_csv = '/home/majinrong/data/images/_train.csv'
train_images = '/home/majinrong/data/images/'

val_csv = '/home/majinrong/data/images/_val.csv'
val_images = '/home/majinrong/data/images/'

transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # 转换为三通道
    transforms.RandomCrop(224, padding=4),  # 裁剪到 32x32，填充 4 个像素
    transforms.Resize(size),  # 调整大小（根据你的需求设置 size）
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 三通道的归一化
])

transform_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # 转换为三通道
    transforms.Resize(size),  # 调整大小（根据你的需求设置 size）
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 三通道的归一化
])

if aug:
    N = 2
    M = 14
    transform_train.transforms.insert(0, RandAugment(N, M))

trainset = CustomImageDataset(csv_file=train_csv, root_dir=train_images, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

valset = CustomImageDataset(csv_file=val_csv, root_dir=val_images, transform=transform_test)
valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, num_workers=8)


# trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)
# testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

# classes = ('t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot')

# Model factory
print('==> Building model..')
if args.net == 'res18':
    net = ResNet18()
elif args.net == 'vgg':
    net = VGG('VGG19')
elif args.net == 'res34':
    net = ResNet34()
elif args.net == 'res50':
    net = ResNet50()
elif args.net == 'res101':
    net = ResNet101()
elif args.net == "convmixer":
    net = ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=10)
elif args.net == "mlpmixer":
    from models.mlpmixer import MLPMixer
    net = MLPMixer(
        image_size=224,
        channels=3,
        patch_size=args.patch,
        dim=512,
        depth=6,
        num_classes=185
    )
elif args.net == "vit_small":
    from models.vit_small import ViT
    net = ViT(
        image_size=size,
        patch_size=args.patch,
        num_classes=185,
        dim=int(args.dimhead),
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1
    )
elif args.net == "vit_tiny":
    from models.vit_small import ViT
    net = ViT(
        image_size=size,
        patch_size=args.patch,
        num_classes=185,
        dim=int(args.dimhead),
        depth=4,
        heads=6,
        mlp_dim=256,
        dropout=0.1,
        emb_dropout=0.1
    )
elif args.net == "simplevit":
    from models.simplevit import SimpleViT
    net = SimpleViT(
        image_size=size,
        patch_size=args.patch,
        num_classes=185,
        dim=int(args.dimhead),
        depth=6,
        heads=8,
        mlp_dim=512
    )
elif args.net == "vit":
    net = ViT(
        image_size=size,
        patch_size=args.patch,
        num_classes=185,
        dim=int(args.dimhead),
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1
    )
elif args.net == "vit_timm":
    import timm
    net = timm.create_model("vit_base_patch16_384", pretrained=True)
    net.head = nn.Linear(net.head.in_features, 185)
elif args.net == "cait":
    from models.cait import CaiT
    net = CaiT(
        image_size=size,
        patch_size=args.patch,
        num_classes=185,
        dim=int(args.dimhead),
        depth=6,
        cls_depth=2,
        heads=8,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1,
        layer_dropout=0.05
    )
elif args.net == "cait_small":
    from models.cait import CaiT
    net = CaiT(
        image_size=size,
        patch_size=args.patch,
        num_classes=10,
        dim=int(args.dimhead),
        depth=6,
        cls_depth=2,
        heads=6,
        mlp_dim=256,
        dropout=0.1,
        emb_dropout=0.1,
        layer_dropout=0.05
    )
elif args.net == "swin":
    from models.swin import swin_t
    net = swin_t(window_size=args.patch,
                 num_classes=10,
                 downscaling_factors=(2,2,2,1))
elif args.net == "mobilevit":
    net = mobilevit_xxs(size, 10)
else:
    raise ValueError(f"'{args.net}' is not a valid model")

# 先将模型移动到主设备 GPU 5，再使用 DataParallel
net.to(device)
if torch.cuda.is_available() and args.dp:
    print("主设备为:", device)
    print("使用数据并行，设备列表: [5, 6]")
    net = torch.nn.DataParallel(net, device_ids=[1,4])
    cudnn.benchmark = True

print(f"当前使用的主设备: {device} - {torch.cuda.get_device_name(device)}")

if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}-ckpt.t7'.format(args.net))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar(batch_idx, len(trainloader), 
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):  # 使用 valloader 进行验证
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(valloader), 
                         'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {"model": net.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict()}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+args.net+'-{}-ckpt.t7'.format(args.patch))
        best_acc = acc

    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {acc:.5f}'
    print(content)
    with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc

list_loss = []
list_acc = []
if usewandb:
    wandb.watch(net)

for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloss = train(epoch)
    val_loss, acc = test(epoch)
    scheduler.step(epoch-1)
    list_loss.append(val_loss)
    list_acc.append(acc)
    if usewandb:
        wandb.log({'epoch': epoch, 'train_loss': trainloss, 'val_loss': val_loss, "val_acc": acc,
                   "lr": optimizer.param_groups[0]["lr"], "epoch_time": time.time()-start})
    with open(f'log/log_{args.net}_patch{args.patch}.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list_loss) 
        writer.writerow(list_acc) 
    print(list_loss)

if usewandb:
    wandb.save("wandb_{}.h5".format(args.net))
