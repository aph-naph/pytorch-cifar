'''Train MNIST with PyTorch.'''
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
from data_utils import get_data_loaders
from utils import progress_bar

from datetime import datetime

import logging
import random # only used for seeding
import numpy as np # only used for seeding

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
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
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

val_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

batch_size = 128
val_split = 0.1
num_workers = 8

train_loader, val_loader, test_loader = get_data_loaders(
    dataset=torchvision.datasets.MNIST,
    data_dir='./data',
    train_transform=train_transform,
    val_transform=val_test_transform,
    test_transform=val_test_transform,
    batch_size=batch_size,
    val_split=val_split,
    num_workers=num_workers
)

classes = tuple(range(10))

# Model
print('\n==> Building model..')
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

act_fns = [nn.ReLU, nn.LeakyReLU, MishAuto, SwishAuto, ProposedAuto, nn.Sigmoid, nn.GELU, nn.Tanh]
# net, model_name = BasicConvResModel(len(classes), nn.ReLU)
net, model_name = BasicConvModel(1, len(classes), act_fns[args.act_fn - 1])
ds_name = 'MNIST'
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
    checkpoint = torch.load('./checkpoint/ckpt.pt')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

fmt = "%d-%m-%Y-%H:%M:%S"
curr_time = lambda: datetime.strftime(datetime.now(), fmt)
ckpt_dir = model_name
start_time = curr_time()

if not os.path.isdir('./logs/{}'.format(ds_name)):
    os.mkdir('./logs/{}'.format(ds_name))

logfile = "./logs/{}/{}-{}-epochs-{}.log".format(ds_name, model_name, start_time, total_epochs)
logging.basicConfig(filename=logfile, level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p - ')
logging.info("Logger initialized.")

for message in [
    "Model: " + model_name,
    "Train config: " + '{}-epochs-{}'.format(model_name, total_epochs),
    "Training start time: " + curr_time()
]:
    print(message)
    logging.info(message)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.RMSprop(net.parameters(), args.lr)
                          # momentum=0.9, weight_decay=5e-4)
optimizer = optim.SGD(net.parameters(), lr=args.lr)
#                        momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
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
        progress_bar(batch_idx, len(train_loader), msg) 

        if batch_idx + 1 == len(train_loader):
            logging.info('Epoch {} | Train Batch: {}/{} | {}'.format(epoch, batch_idx + 1, len(train_loader), msg))

def evaluate(loader):
    net.eval()
    eval_loss = 0
    correct = 0
    total = 0
    msg = ''

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            eval_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            msg = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (eval_loss/(batch_idx+1), 100.*correct/total, correct, total)
            progress_bar(batch_idx, len(loader), msg)

    return eval_loss, correct, total, msg

def val(epoch):
    global best_acc
    val_loss, correct, total, msg = evaluate(val_loader)
    val_len = len(val_loader)
    logging.info('Epoch {} | Val Batch: {}/{} | {}'.format(epoch, val_len, val_len, msg))

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

        if not os.path.isdir('checkpoint/{}'.format(ds_name)):
            os.mkdir('checkpoint/{}'.format(ds_name))

        if not os.path.isdir('checkpoint/{}/{}'.format(ds_name, ckpt_dir)):
            os.mkdir('checkpoint/{}/{}'.format(ds_name, ckpt_dir))

        if not os.path.isdir('checkpoint/{}/{}/{}'.format(ds_name, ckpt_dir, start_time)):
            os.mkdir('checkpoint/{}/{}/{}'.format(ds_name, ckpt_dir, start_time))

        torch.save(
            state, 
            './checkpoint/{}/{}/{}/{}.pt'.format(ds_name, ckpt_dir, start_time, curr_time()))
        best_acc = acc

def test():
    test_loss, correct, total, msg = evaluate(test_loader)
    test_len = len(test_loader)
    logging.info('Epoch {} | Test Batch: {}/{} | {}'.format(epoch, test_len, test_len, msg))

for epoch in range(total_epochs):
    train(epoch)
    val(epoch)

print("\nEvaluating on test set...")
test()
