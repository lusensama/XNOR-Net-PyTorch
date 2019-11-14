
import torch
import os
import torch.nn as nn
import math


class ConvReLU(nn.Module): # change the name of BinConv2d
    def __init__(self, input_channels, output_channels, dropout=0.0,):
        super(ConvReLU, self).__init__()
        self.layer_type = 'ConvReLU'
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        if self.dropout!=0:
            x = self.dropout(x)
        return x


class Vgg(nn.Module):

    def __init__(self, dr=0.1, num_classes=1000):
        super(Vgg, self).__init__()
        self.dropout_ratio = dr
        self.features = nn.Sequential(
            ConvReLU(3, 64, dropout=0.1),           # ConvReLU(3,64):add(nn.Dropout(0.1))
            ConvReLU(64, 64),                       # ConvReLU(64,64)
            nn.AvgPool2d(kernel_size=2, stride=2),  # model:add(Avg(2,2,2,2))
            #
            ConvReLU(64, 128, dropout=0.1),         # ConvReLU(64,128):add(nn.Dropout(0.1))
            ConvReLU(128, 128),                     # ConvReLU(128,128)
            nn.AvgPool2d(kernel_size=2, stride=2),  # model:add(Avg(2,2,2,2))
            #
            ConvReLU(128, 256, dropout=0.1),        # ConvReLU(128,256):add(nn.Dropout(0.1))
            ConvReLU(256, 256, dropout=0.1),        # ConvReLU(256,256):add(nn.Dropout(0.1))
            ConvReLU(256, 256),                     # ConvReLU(256,256)
            nn.AvgPool2d(kernel_size=2, stride=2),  # model:add(Avg(2,2,2,2):ceil())
            #
            ConvReLU(256, 512, dropout=0.1),        # ConvReLU(256,512):add(nn.Dropout(0.1))
            ConvReLU(512, 512, dropout=0.1),        # ConvReLU(512,512):add(nn.Dropout(0.1))
            ConvReLU(512, 512),                     # ConvReLU(512,512)
            nn.AvgPool2d(kernel_size=2, stride=2),  # model:add(Avg(2,2,2,2):ceil())
            #
            ConvReLU(512, 512, dropout=0.1),        # ConvReLU(512,512):add(nn.Dropout(0.1))
            ConvReLU(512, 512, dropout=0.1),        # ConvReLU(512,512):add(nn.Dropout(0.1))
            ConvReLU(512, 512),                     # ConvReLU(512,512)
            nn.AvgPool2d(kernel_size=2, stride=2),  # model:add(Avg(2,2,2,2):ceil())
        )
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_ratio),         # classifier:add(nn.Dropout(0.1))
            nn.Linear(512 * 7 * 7, 4096, bias=False),           # classifier:add(nn.Linear(512,512,false))
            nn.ReLU(True),                          # classifier:add(ReLU(true))
            nn.Dropout(self.dropout_ratio),         # classifier:add(nn.Dropout(0.1))
            nn.Linear(4096, 4096, bias=False),                  # classifier:add(nn.Linear(512,10,false))
            nn.ReLU(True),                          # classifier:add(ReLU(true))
            nn.Dropout(self.dropout_ratio),         # additional dropout
            nn.Linear(4096, num_classes, bias=False),           # additional dense layer
        )
        # self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)                    # model:add(nn.View(512))
        x = self.classifier(x)
        return x


def vgg_net(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Vgg(**kwargs)
    if pretrained:
        model_path = 'model_list/vgg_best_full_prec.pth.tar'
        # model_path = 'alexnet_XNOR_cpu.pth'
        pretrained_model = torch.load(model_path)
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in pretrained_model.items():
        #     name = k.replace(".module", "")  # remove `module.`
        #     new_state_dict[name] = v
        # load params
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
        model.load_state_dict(pretrained_model['state_dict'])
        # model.load_state_dict(pretrained_model['state_dict'], strict=True)
    return model