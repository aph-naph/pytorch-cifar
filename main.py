'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from activations_autofn import *
from utils import progress_bar

from datetime import datetime

import logging
import random # only used for seeding
import numpy as np # only used for seeding

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--act_fn', default=1, type=int, help='activation function')
parser.add_argument('--max_epochs', default=50, type=int, help='epochs')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
# net = net.to(device)

act_fns = [nn.ReLU, nn.LeakyReLU, MishAuto, SwishAuto, ProposedAuto]
# net, model_name = BasicConvResModel(len(classes), nn.ReLU)
net, model_name = BasicConvModel(len(classes), act_fns[args.act_fn - 1])

max_epochs = args.max_epochs
total_epochs = max_epochs - start_epoch

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    print("CUDA Enabled")

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)


fmt = "%d-%m-%Y-%H:%M:%S"
curr_time = lambda: datetime.strftime(datetime.now(), fmt)
ckpt_dir = model_name
start_time = curr_time()

logfile = "./logs/{}-{}-epochs-{}.log".format(model_name, start_time, total_epochs)
logging.basicConfig(filename=logfile, level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p - ')
logging.info("Logger initialized.")

for message in [
    "Model: " + model_name,
    "Train config: " + '{}-epochs-{}'.format(model_name, total_epochs),
    "Training start time: " + curr_time()
]:
    print(message)
    logging.info(message)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        msg = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total)
        progress_bar(batch_idx, len(trainloader), msg) 

        if batch_idx + 1 == len(trainloader):
            logging.info('Epoch {} | Train Batch: {}/{} | {}'.format(epoch, batch_idx + 1, len(trainloader), msg))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            msg = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total)
            progress_bar(batch_idx, len(testloader), msg)

            if batch_idx + 1 == len(trainloader):
                logging.info('Epoch {} | Train Batch: {}/{} | {}'.format(epoch, batch_idx + 1, len(trainloader), msg))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')

        if not os.path.isdir('checkpoint/{}'.format(ckpt_dir)):
            os.mkdir('checkpoint/{}'.format(ckpt_dir))

        if not os.path.isdir('checkpoint/{}/{}'.format(ckpt_dir, start_time)):
            os.mkdir('checkpoint/{}/{}'.format(ckpt_dir, start_time))

        torch.save(
            state, 
            './checkpoint/{}/{}/{}.pth'.format(ckpt_dir, start_time, curr_time()))
        best_acc = acc


for epoch in range(total_epochs):
    train(epoch)
    test(epoch)
    scheduler.step()
