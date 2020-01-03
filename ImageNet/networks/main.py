import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
# import torchvision.datasets as datasets
import model_list
import util
from data_loader import load_data
import matplotlib as plt
import numpy as np
# set the seed
torch.manual_seed(1)
torch.cuda.manual_seed(1)

import sys
import gc
#--evaluate --pretrained
# --data C:\imagenet_data --epochs 50  --workers 4 -b 128 --lr 0.05 --resume model_best_56.pth.tar
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='alexnet',
                    help='model architecture (default: alexnet)')
parser.add_argument('--data', metavar='DATA_PATH', default='./data/',
                    help='path to imagenet data (default: ./data/)')
parser.add_argument('--caffe-data',  default=False, action='store_true',
                    help='whether use caffe-data')
parser.add_argument('--cifar',  default=False, action='store_true',
                    help='use cifar data by default')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.90, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-5)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', action='store', default=None,
                    help='the path to the pretrained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--workdir', action='store', default=None,
                    help='the path to store everything')

best_prec1 = 0

# define global bin_op
bin_op = None

# cuda0 = torch.device('cuda:0')
# cuda1 = torch.device('cuda:1')

def main():

    global args, best_prec1
    args = parser.parse_args()

    # create model
    if args.arch=='alexnet':
        model = model_list.alexnet(pretrained=args.pretrained)
        input_size = 224
    elif args.arch=='vgg16':
        model = model_list.vgg_net(pretrained=args.pretrained)
        input_size = 224
    elif args.arch=='vgg15_bwn':
        model = model_list.vgg_15(pretrained=args.pretrained)
        input_size = 224
    elif args.arch=='vgg15_bn_XNOR':
        model = model_list.vgg15_bn_XNOR(pretrained=args.pretrained)
        input_size = 224
    elif args.arch=='vgg15ab':
        model = model_list.vgg15ab(pretrained=args.pretrained)
        input_size = 224
    else:
        raise Exception('Model not supported yet')

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        pass
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                 # betas=(0.0, 0.999),

                                weight_decay=args.weight_decay)
# scratch
#     for m in model.modules():
#         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#             c = float(m.weight.data[0].nelement())
#             m.weight.data = m.weight.data.normal_(0, 2.0/c)
#         elif isinstance(m, nn.BatchNorm2d):
#             m.weight.data = m.weight.data.zero_().add(1.0)
#             m.bias.data = m.bias.data.zero_()

    # optionally resume from a checkpoint
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(args.resume)
    #         # TODO: Temporary remake
    #         # args.start_epoch = 0
    #         # best_prec1 = 0.0
    #         # model.features = torch.nn.DataParallel(model.features)
    #         try:
    #             args.start_epoch = checkpoint['epoch']
    #             best_prec1 = checkpoint['best_prec1']
    #
    #             model.load_state_dict(checkpoint['state_dict'])
    #         except KeyError:
    #             model.load_state_dict(checkpoint)
    #             pass
    #
    #
    #
    #
    #         # optimizer.load_state_dict(checkpoint['optimizer'])
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(args.resume, args.start_epoch))
    #         del checkpoint
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code

    if args.caffe_data:
        print('==> Using Caffe Dataset')
        cwd = os.getcwd()
        sys.path.append(cwd+'/../')
        import datasets as datasets
        import datasets.transforms as transforms
        if not os.path.exists(args.data+'/imagenet_mean.binaryproto'):
            print("==> Data directory"+args.data+"does not exits")
            print("==> Please specify the correct data path by")
            print("==>     --data <DATA_PATH>")
            return

        normalize = transforms.Normalize(
                meanfile=args.data+'/imagenet_mean.binaryproto')


        train_dataset = datasets.ImageFolder(
            args.data,
            transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                transforms.RandomSizedCrop(input_size),
            ]),
            Train=True)

        train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(args.data, transforms.Compose([
                transforms.ToTensor(),
                normalize,
                transforms.CenterCrop(input_size),
            ]),
            Train=False),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    elif args.cifar:
        import torchvision.transforms as transforms
        import torchvision
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        val_loader = torch.utils.data.DataLoader(testset, batch_size=100,
                                                 shuffle=False, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    else:
        print('==> Using Pytorch Dataset')
        import torchvision
        import torchvision.transforms as transforms
        import torchvision.datasets as datasets
        # traindir = os.path.join(args.data, 'train')
        # valdir = os.path.join(args.data, 'test')
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        if True:
        #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        # else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size//2 if args.arch.startswith('vgg') else args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    # print (model)

    # define the binarization operator
    global bin_op
    bin_op = util.BinOp(model)

    if args.evaluate:
        # bin_op.binarization()
        # save_checkpoint(model.state_dict(), False, 'vgg_binarized')
        # bin_op.restore()
        validate(val_loader, model, criterion)
        return
    val_prec_list = []
    writer = SummaryWriter(args.workdir+'/runs/loss_graph')
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, writer)

        # evaluate on validation set
        prec1, prec5 = validate(val_loader, model, criterion)
        val_prec_list.append(prec1)
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filename='{}/{}_'.format(args.workdir, args.arch))
        writer.add_scalar('top1 accuracy', prec1, epoch)
        writer.add_scalar('top5 accuracy', prec5, epoch)
        writer.add_scalar('learning rate', args.lr, epoch)
    print(val_prec_list)
# def main():
#
#     global args, best_prec1
#     args = parser.parse_args()
#
#     # create model
#     if args.arch=='alexnet':
#         model = model_list.alexnet(pretrained=args.pretrained)
#         input_size = 227
#     else:
#         raise Exception('Model not supported yet')
#
#     if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
#         model.features = torch.nn.DataParallel(model.features)
#         model.cuda()
#     else:
#         model = torch.nn.DataParallel(model).cuda()
#
#     # define loss function (criterion) and optimizer
#     criterion = nn.CrossEntropyLoss().cuda()
#
#     optimizer = torch.optim.Adam(model.parameters(), args.lr,
#                                 weight_decay=args.weight_decay)
#
#     for m in model.modules():
#         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#             c = float(m.weight.data[0].nelement())
#             m.weight.data = m.weight.data.normal_(0, 2.0/c)
#         elif isinstance(m, nn.BatchNorm2d):
#             m.weight.data = m.weight.data.zero_().add(1.0)
#             m.bias.data = m.bias.data.zero_()
#
#     # optionally resume from a checkpoint
#     if args.resume:
#         if os.path.isfile(args.resume):
#             print("=> loading checkpoint '{}'".format(args.resume))
#             checkpoint = torch.load(args.resume)
#             # TODO: Temporary remake
#             args.start_epoch = 0
#             best_prec1 = 0.0
#             model.load_state_dict(checkpoint, strict=False)
#             # args.start_epoch = checkpoint['epoch']
#             # best_prec1 = checkpoint['best_prec1']
#             # model.load_state_dict(checkpoint['state_dict'])
#             # optimizer.load_state_dict(checkpoint['optimizer'])
#             print("=> loaded checkpoint '{}' (epoch {})"
#                   .format(args.resume, args.start_epoch))
#             del checkpoint
#         else:
#             print("=> no checkpoint found at '{}'".format(args.resume))
#
#     cudnn.benchmark = True
#
#     # Data loading code
#
#     if args.caffe_data:
#         print('==> Using Caffe Dataset')
#         cwd = os.getcwd()
#         sys.path.append(cwd+'/../')
#         import datasets as datasets
#         import datasets.transforms as transforms
#         if not os.path.exists(args.data+'/imagenet_mean.binaryproto'):
#             print("==> Data directory"+args.data+"does not exits")
#             print("==> Please specify the correct data path by")
#             print("==>     --data <DATA_PATH>")
#             return
#
#         normalize = transforms.Normalize(
#                 meanfile=args.data+'/imagenet_mean.binaryproto')
#
#
#         train_dataset = datasets.ImageFolder(
#             args.data,
#             transforms.Compose([
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 normalize,
#                 transforms.RandomSizedCrop(input_size),
#             ]),
#             Train=True)
#
#         train_sampler = None
#
#         train_loader = torch.utils.data.DataLoader(
#             train_dataset, batch_size=args.batch_size, shuffle=False,
#             num_workers=args.workers, pin_memory=True, sampler=train_sampler)
#
#         val_loader = torch.utils.data.DataLoader(
#             datasets.ImageFolder(args.data, transforms.Compose([
#                 transforms.ToTensor(),
#                 normalize,
#                 transforms.CenterCrop(input_size),
#             ]),
#             Train=False),
#             batch_size=args.batch_size, shuffle=False,
#             num_workers=args.workers, pin_memory=True)
#     elif args.cifar:
#         import torchvision.transforms as transforms
#         import torchvision
#         transform = transforms.Compose(
#             [transforms.ToTensor(),
#              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
#         trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                                 download=True, transform=transform)
#         train_loader = torch.utils.data.DataLoader(trainset, batch_size=128,
#                                                   shuffle=True, num_workers=2)
#
#         testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                                download=True, transform=transform)
#         val_loader = torch.utils.data.DataLoader(testset, batch_size=100,
#                                                  shuffle=False, num_workers=2)
#
#         classes = ('plane', 'car', 'bird', 'cat',
#                    'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#
#
#     else:
#         print('==> Using Pytorch Dataset')
#         import torchvision
#         import torchvision.transforms as transforms
#         import torchvision.datasets as datasets
#         # traindir = os.path.join(args.data, 'train')
#         # valdir = os.path.join(args.data, 'test')
#         train_loader, val_loader, test_loader, input_shape = load_data(dataset='imagenet', data_dir=args.data, batch_size=args.batch_size, workers=4)
#
#
#     # print (model)
#
#     # define the binarization operator
#     global bin_op
#     bin_op = util.BinOp(model)
#
#     if args.evaluate:
#         validate(val_loader, model, criterion)
#         return
#     val_prec_list = []
#     for epoch in range(args.start_epoch, args.epochs):
#         adjust_learning_rate(optimizer, epoch)
#
#         # train for one epoch
#         train(train_loader, model, criterion, optimizer, epoch)
#
#         # evaluate on validation set
#         prec1 = validate(val_loader, model, criterion)
#         val_prec_list.append(prec1)
#         # remember best prec@1 and save checkpoint
#         is_best = prec1 > best_prec1
#         best_prec1 = max(prec1, best_prec1)
#         save_checkpoint({
#             'epoch': epoch + 1,
#             'arch': args.arch,
#             'state_dict': model.state_dict(),
#             'best_prec1': best_prec1,
#             'optimizer' : optimizer.state_dict(),
#         }, is_best)
#     print(val_prec_list)

def train(train_loader, model, criterion, optimizer, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    loss_record = 0.0
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        timers = time.time()
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # process the weights including binarization
        bin_op.binarization()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        loss_record += loss.item()
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # restore weights
        bin_op.restore()
        bin_op.updateBinaryGradWeight()

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            writer.add_scalar('training loss',
                              loss_record/10,
                            epoch * len(train_loader) + i)

            # 'timer {times}'.format(
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            loss_record = 0.0
        gc.collect()



def validate(val_loader, model, criterion):
    print("start validation")
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    # model.train()
    model.eval()
    end = time.time()
    bin_op.binarization()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
    bin_op.restore()

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='vgg15'):
    torch.save(state, filename + 'checkpoint.pth.tar')
    if is_best:
        shutil.copyfile(filename + 'checkpoint.pth.tar', filename +'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr
    update_list = [120, 200, 240, 280]
    print ('Learning rate:', lr)
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
