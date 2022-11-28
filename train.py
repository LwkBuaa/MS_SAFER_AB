from __future__ import print_function
import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from dataset import CKDataset, FERPlusDataset, RAFDBDataset
from utils import *
from tensorboardX import SummaryWriter
import gc
from nets import SpResNet18


def adjust_lr(epoch, optimizer):
    if epoch < 20:
        lr = args.lr * (epoch + 1) / 20
    elif 20 <= epoch < 50:
        lr = args.lr
    elif 50 <= epoch < 80:
        lr = args.lr * 0.1
    elif 80 <= epoch:
        lr = args.lr * 0.01
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = 0.1 * lr
    optimizer.param_groups[2]['lr'] = lr
    optimizer.param_groups[3]['lr'] = lr
    return lr


def train(epoch):
    cur_lr = adjust_lr(epoch, optimizer)
    train_loss = AverageMeter()
    train_time = AverageMeter()  # one batch train time
    correct = 0
    total = 0

    model.train()
    start_time = time.time()

    for batch_idx, (inputs, labels) in enumerate(train_dataloader):
        gc.collect()
        torch.cuda.empty_cache()
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        out0 = model(inputs)
        labels = labels.clone().detach()
        labels = torch.tensor(labels, dtype=torch.float32)
        loss = criterion_cls(out0, labels.long())
        _, predicted = out0.max(1)
        correct += (predicted.eq(labels).sum().item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), inputs.size(0))
        total += inputs.size(0)

        train_time.update(time.time()-start_time)
        start_time = time.time()
        if batch_idx % 2 == 0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {train_time.val:.3f} ({train_time.avg:.3f}) '
                  'lr:{:.3f} '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'Acc: {:.2f}'.format(
                epoch, batch_idx, len(train_dataset), cur_lr, 100. * correct / total,
                train_time=train_time, train_loss=train_loss))


def test(epoch):
    test_loss = AverageMeter()
    test_time = AverageMeter()  # one batch test time
    correct = 0
    total = 0
    model.eval()
    start_time = time.time()
    for batch_idx, (inputs, labels) in enumerate(test_dataloader):
        gc.collect()
        torch.cuda.empty_cache()
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        out0 = model(inputs)
        labels = labels.clone().detach()
        labels = torch.tensor(labels, dtype=torch.float32)
        loss = criterion_cls(out0, labels.long())
        _, predicted = out0.max(1)
        correct += (predicted.eq(labels).sum().item())
        test_loss.update(loss.item(), inputs.size(0))
        total += inputs.size(0)

        test_time.update(time.time() - start_time)
        start_time = time.time()
        if batch_idx % 2 == 0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {test_time.val:.3f} ({test_time.avg:.3f}) '
                  'Loss: {test_loss.val:.4f} ({test_loss.avg:.4f}) '
                  'Acc: {:.2f}'.format(
                epoch, batch_idx, len(test_dataset), 100. * correct / total,
                test_time=test_time, test_loss=test_loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='FERPlus', help='dataset name: regdb or sysu]')
    parser.add_argument('--epochs', default=100, type=int, help='the num of train epoch')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate, 0.00035 for adam')
    parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
    parser.add_argument('--backbone', default='ResNet18', type=str, help='network baseline: resnet50')
    parser.add_argument('--method', default='SpResNet18', type=str, help='network baseline: resnet50')
    parser.add_argument('--log_path', default='log', type=str, help='log save path')
    parser.add_argument('--workers', default=4, type=int, metavar='N', help='num of data load workers (default: 0)')
    parser.add_argument('--img_size', default=(48, 48), type=tuple, metavar='imgw', help='img width')
    parser.add_argument('--batch_train', default=32, type=int, metavar='B', help='training batch size')
    parser.add_argument('--batch_test', default=2, type=int, metavar='tb', help='testing batch size')
    parser.add_argument('--gpu', default='0', type=str, help='gpu device id for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--seed', default=0, type=int, metavar='t', help='random seed')
    args = parser.parse_args()
    print("==========\nArgs:{}".format(args))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)
    set_seed(args.seed)

    if args.dataset == 'FERPlus':
        data_path = r"DataSet\FERPlus\data\emotion_cls"
        log_path = args.log_path + '/FER_log/'
    elif args.dataset == 'CK+':
        data_path = r"DataSet\CK+\emotion_cls"
        log_path = args.log_path + '/CK_log/'
    elif args.dataset == 'RAF-DB':
        data_path = r"DataSet\RAF-DB"
        log_path = args.log_path + '/RAFDB_log/'
    else:
        print("==========\nDataset Error(FERPlus | CK+ | RAF-DB)\n==========")
    if not os.path.isdir(log_path):
        os.makedirs(log_path)

    suffix = '{}_{}_lr_{}_seed_{}'.format(args.backbone, args.dataset, args.lr, args.seed)
    sys.stdout = Logger(log_path + suffix + '.txt')

    print('==========\n>>> Loading data...')
    start = time.time()
    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Pad(10),
        transforms.RandomCrop(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
        normalize,
    ])

    if args.dataset == 'FERPlus':
        train_dataset = FERPlusDataset(data_dir=data_path, transform=transform_train)
        test_dataset = FERPlusDataset(data_dir=data_path, transform=transform_train, mode='test')
    elif args.dataset == 'CK+':
        train_dataset = CKDataset(data_dir=data_path, transform=transform_train)
        test_dataset = CKDataset(data_dir=data_path, transform=transform_train, mode='test')
    elif args.dataset == 'RAF-DB':
        train_dataset = RAFDBDataset(data_dir=data_path, transform=transform_train)
        test_dataset = RAFDBDataset(data_dir=data_path, transform=transform_train, mode='test')
    else:
        train_dataset = []
        test_dataset = []
        print("==========\nDataset Error(FERPlus | CK+ | RAF-DB)")
    print('==========\nDataset {} statistics:'.format(args.dataset))
    print('subset | # n_label | # images')
    print('Train  |  {:5d}    | {:8d}'.format(len(train_dataset.labels), len(train_dataset)))
    print('Test   |  {:5d}    | {:8d}'.format(len(train_dataset.labels), len(test_dataset)))
    print('Data Loading Time:\t {:.3f}'.format(time.time() - start))

    print('==========\n>>> Build model...')
    if args.method == 'SpResNet18':
        model = SpResNet18(pretrained=True)
    else:
        print("==========\nModel Choice Error(SpResNet)")
    model.to(device)
    cudnn.benchmark = True

    print('==========\n>>> Define loss function...')
    criterion_cls = nn.CrossEntropyLoss()
    criterion_cls.to(device)
    print('==========\n>>> Define optimizer...')
    ignore_params = list(map(id, model.conv1.parameters())) + \
                    list(map(id, model.fc1.parameters())) + \
                    list(map(id, model.fc2.parameters()))
    base_params = filter(lambda p: id(p) not in ignore_params, model.parameters())
    optimizer = optim.SGD([
        {'params': model.conv1.parameters(), 'lr': args.lr},
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': model.fc1.parameters(), 'lr': args.lr},
        {'params': model.fc2.parameters(), 'lr': args.lr}
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    print('==========\n>>> Start training...')
    best_acc = 0  # best test accuracy
    for epoch in range(args.epochs):
        train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_train, num_workers=args.workers, drop_last=True)
        train(epoch)

        if epoch > 0 and epoch % 2 == 0:
            print('==========\nTest epoch: {}'.format(epoch))
            test_dataloader = data.DataLoader(test_dataset, batch_size=args.batch_test, num_workers=args.workers, drop_last=True)
            acc = test(epoch)
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                state = {
                    'model': model.state_dict(),
                    'epoch': best_epoch,
                    'acc': best_acc
                }
                torch.save(state, args.log_path + suffix + '_best.t')
                print('Epoch: {:4d} | Acc: {:4%}'.format(best_epoch, best_epoch))
