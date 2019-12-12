import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import sys
import os
import torch
import argparse
import util
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import models

# writer = SummaryWriter('runs/pretrained')
def save_state(model, best_acc, arch, pretrained):

    print('==> Saving model ...')
    state = {
        'best_acc': best_acc,
        'state_dict': model.state_dict(),
    }
    # for key in state['state_dict'].keys():
    #     if 'module' in key:
    #         state['state_dict'][key.replace('module.', '')] = \
    #             state['state_dict'].pop(key)
    torch.save(state, '{}/{}_cifar100_from_{}.pth.tar'.format(args.workdir, arch, 'pretrained' if pretrained else 'scratch'))


def train(epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(trainloader):
        # process the weights including binarization
        bin_op.binarization()

        # forwarding
        data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output = model(data)

        # backwarding
        loss = criterion(output, target)
        loss.backward()

        # restore weights
        bin_op.restore()
        bin_op.updateBinaryGradWeight()

        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                       100. * batch_idx / len(trainloader), loss.data.item(),
                optimizer.param_groups[0]['lr']))

    return

# vgg15m_cifar100model_best67.7.pth.tar
def evaluate(save_binary = False):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    bin_op.binarization()
    for data, target in testloader:
        data, target = Variable(data.cuda()), Variable(target.cuda())

        output = model(data)
        test_loss += criterion(output, target).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    if save_binary:
        save_state(model, best_acc, 'binarized_'+args.arch, args.pretrained)
    bin_op.restore()
    # if acc > best_acc:
    #     best_acc = acc
    #     save_state(model, best_acc, args.arch, args.pretrained)
    # else:
    #     save_state(model, best_acc, 'checkpoint_' + args.arch, args.pretrained)
    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f} , Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * 128., correct, len(testloader.dataset),
        100. * float(correct) / len(testloader.dataset)))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc))

def test(epc, writer):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    bin_op.binarization()
    for data, target in testloader:
        data, target = Variable(data.cuda()), Variable(target.cuda())

        output = model(data)
        test_loss += criterion(output, target).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    bin_op.restore()
    acc = 100. * float(correct) / len(testloader.dataset)

    if acc > best_acc:
        best_acc = acc
        save_state(model, best_acc, args.arch, args.pretrained)
    else:
        save_state(model, best_acc, 'checkpoint_'+ args.arch, args.pretrained)
    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f} , Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * 128., correct, len(testloader.dataset),
        100. * float(correct) / len(testloader.dataset)))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc))
    writer.add_scalar('{}_average_loss'.format(args.arch), test_loss, epc)
    writer.add_scalar('{}_test_accuracy'.format(args.arch), 100. * float(correct) / len(testloader.dataset), epc)
    writer.add_scalar('{}_learning_rate'.format(args.arch), float(args.lr), epc)

    return


def adjust_learning_rate(optimizer, epoch):
    update_list = [30, 60, 90, 120, 150, 180]
    # update_list = [81, 122]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5
            param_group['weight_decay'] = 0
    return


if __name__ == '__main__':
    # prepare the options
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true',
                        help='set if only CPU is available')
    parser.add_argument('--data', action='store', default='./data/',
                        help='dataset path')
    parser.add_argument('--arch', action='store', default='alexnet',
                        help='the architecture for the network: nin')
    parser.add_argument('--lr', action='store', default='0.01',
                        help='the intial learning rate')
    parser.add_argument('--pretrained', action='store', default=None,
                        help='the path to the pretrained model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--workdir', action='store', default=None,
                        help='the path to store everything')
    args = parser.parse_args()
    print('==> Options:', args)

    # set the seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    if args.pretrained:
        writer = SummaryWriter(args.workdir + '/' +'runs/pretrained')
    else:
        writer = SummaryWriter(args.workdir + '/' + 'runs/scratch')
    # # prepare the data
    # if not os.path.isfile(args.data + '/train_data'):
    #     # check the data path
    #     raise Exception \
    #         ('Please assign the correct data path with --data <DATA_PATH>')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    trainset = torchvision.datasets.CIFAR100('./data', train=True,
                                             transform=transforms.Compose([
                                                 transforms.RandomCrop(32, padding=4),
                                                 transforms.RandomHorizontalFlip(0.5),
                                                 transforms.ToTensor(),
                                                 normalize, ]),
                                             target_transform=None, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100('./data', train=False,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                normalize, ]),
                                            target_transform=None, download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=2)

    # define classes
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # define the model
    print('==> building model', args.arch, '...')
    if args.arch == 'alexnet_bwn':
        model = models.alexnet_bwn()
    elif args.arch == 'alexnet_xnor':
        model = models.alexnet_xnor()
    elif args.arch == 'vgg15_max':
        model = models.vgg_15_max()
    elif args.arch == 'vgg15_avg':
        model = models.vgg_15_avg()
    elif args.arch == 'vgg15_avg_before':
        model = models.vgg_15_avg2()
    elif args.arch == 'vgg15_xnor':
        model = models.vgg_15_xnor()
    else:
        raise Exception(args.arch + ' is currently not supported')

    # initialize the model
    if not args.pretrained:
        print('==> Initializing model parameters ...')
        best_acc = 0
        model.features = torch.nn.DataParallel(model.features)
        # model = torch.nn.DataParallel(model)
        model.cuda('cuda:0')
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)
                # m.bias.data.zero_()
    else:
        print('==> Load pretrained model form', args.pretrained, '...')
        pretrained_model = torch.load(args.pretrained)
        best_acc = 0
        # try:
        #     best_acc = pretrained_model['best_acc1']
        # except KeyError:
        #     best_acc = pretrained_model['best_acc']


        model.features = torch.nn.DataParallel(model.features)
        # model = torch.nn.DataParallel(model)
        model.cuda('cuda:0')
        model.load_state_dict(pretrained_model['state_dict'])
        # model.cuda('0')
        # model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))



    # if not isinstance(model, nn.DataParallel):
    #     print('============Not an instance of DataPrallel===========')
    #     model.cuda()
    #     model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    print(model)

    # define solver and criterion
    base_lr = float(args.lr)
    param_dict = dict(model.named_parameters())
    params = []

    for key, value in param_dict.items():
        params += [{'params': [value], 'lr': base_lr,
                    'weight_decay': float(args.weight_decay)}]

    optimizer = optim.Adam(params, lr=float(args.lr),
                           # weight_decay=5e-4 # pretrained
                           weight_decay=args.weight_decay # scratch
                           # betas=(0.0, 0.99999)
                           )
    criterion = nn.CrossEntropyLoss()

    # define the binarization operator
    bin_op = util.BinOp(model)

    # do the evaluation if specified
    if args.evaluate:
        evaluate(True)
        exit(0)

    # start training
    for epoch in range(1, 200):
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test(epoch, writer)
