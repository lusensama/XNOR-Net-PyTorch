import torch
import os
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

__all__ = ['AlexNet', 'alexnet']


class BinConv2d(nn.Module): # change the name of BinConv2d
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=-1, padding=-1, groups=1, dropout=0.0,
            Linear=False):
        super(BinConv2d, self).__init__()
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout

        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.Linear = Linear
        if not self.Linear:
            # self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.conv = nn.Conv2d(input_channels, output_channels,
                    kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
        else:
            # self.bn = nn.BatchNorm1d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.linear = nn.Linear(input_channels, output_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x = self.bn(x)
        # binAct = BinActive.apply
        # x = binAct(x)
        # x=BinActive()(x)
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        if not self.Linear:
            x = self.conv(x)
        else:
            x = self.linear(x)
        x = self.relu(x)
        return x



class AlexNet(nn.Module):
    # def __init__(self, num_classes=10):
    #     super(AlexNet, self).__init__()
    #     self.num_classes = num_classes
    #     self.features = nn.Sequential(
    #         nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
    #         # nn.BatchNorm2d(96, eps=1e-4, momentum=0.1, affine=True),
    #         nn.ReLU(inplace=True),
    #         nn.AvgPool2d(kernel_size=2),
    #
    #         BinConv2d(64, 192, kernel_size=3, stride=1, padding=1), # dropout compensate the batch norm
    #         nn.AvgPool2d(kernel_size=2),
    #
    #         BinConv2d(192, 384, kernel_size=3, stride=1, padding=1),
    #
    #         BinConv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=1, dropout=.1),
    #
    #         BinConv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=1, dropout=.1),
    #
    #         nn.AvgPool2d(kernel_size=2, stride=2),
    #     )
    #     self.classifier = nn.Sequential(
    #         BinConv2d(256 * 2 * 2, 4096, Linear=True, dropout=0.1),
    #         BinConv2d(4096, 4096, dropout=0.1, Linear=True),
    #         # nn.BatchNorm1d(4096, eps=1e-3, momentum=0.1, affine=True),
    #         # nn.Dropout(),
    #         nn.Linear(4096, num_classes),
    #     )
    #
    # def forward(self, x):
    #     x = self.features(x)
    #     x = x.view(x.size(0), 256 * 2 * 2)
    #     x = self.classifier(x)
    #     return x
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.dr = .1
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2, bias=False),
            # nn.BatchNorm2d(96, eps=1e-4, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2),
            BinConv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=1),
            nn.AvgPool2d(kernel_size=3, stride=2),
            BinConv2d(256, 384, kernel_size=3, stride=1, padding=1),
            BinConv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=1, dropout=self.dr),
            BinConv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=1, dropout=self.dr),
            nn.AvgPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            BinConv2d(256 * 6 * 6, 4096, Linear=True),
            BinConv2d(4096, 4096, Linear=True, dropout=self.dr),
            # nn.BatchNorm1d(4096, eps=1e-3, momentum=0.1, affine=True),
            # nn.Dropout(),
            nn.Linear(4096, num_classes, bias=False),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model_path = 'model_list/alexnet_checkpoint.pth.tar'
        print('loading pretrianed model at ' + model_path)

        # model_path = 'alexnet_XNOR_cpu.pth'
        pretrained_model = torch.load(model_path)
        model.features = torch.nn.DataParallel(model.features)
        # model.cuda()
        model.load_state_dict(pretrained_model['state_dict'])
        # model.load_state_dict(pretrained_model['state_dict'], strict=True)
    return model
