import torch
import os
import torch.nn as nn
import math


class ConvReLU(nn.Module):  # change the name of BinConv2d
    def __init__(self, input_channels, output_channels, dropout=0.0, ):
        super(ConvReLU, self).__init__()
        self.layer_type = 'ConvReLU'
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        if self.dropout != 0:
            x = self.dropout(x)
        return x


class Vgg(nn.Module):
    def __init__(self, dr=0.1, num_classes=1000):
        super(Vgg, self).__init__()
        self.dropout_ratio = dr
        self.features = nn.Sequential(
            ConvReLU(3, 64, dropout=0.1),  # ConvReLU(3,64):add(nn.Dropout(0.1))
            ConvReLU(64, 64),  # ConvReLU(64,64)
            nn.AvgPool2d(kernel_size=2, stride=2),  # model:add(Avg(2,2,2,2))
            #
            ConvReLU(64, 128, dropout=0.1),  # ConvReLU(64,128):add(nn.Dropout(0.1))
            ConvReLU(128, 128),  # ConvReLU(128,128)
            nn.AvgPool2d(kernel_size=2, stride=2),  # model:add(Avg(2,2,2,2))
            #
            ConvReLU(128, 256, dropout=0.1),  # ConvReLU(128,256):add(nn.Dropout(0.1))
            ConvReLU(256, 256, dropout=0.1),  # ConvReLU(256,256):add(nn.Dropout(0.1))
            ConvReLU(256, 256),  # ConvReLU(256,256)
            nn.AvgPool2d(kernel_size=2, stride=2),  # model:add(Avg(2,2,2,2):ceil())
            #
            ConvReLU(256, 512, dropout=0.1),  # ConvReLU(256,512):add(nn.Dropout(0.1))
            ConvReLU(512, 512, dropout=0.1),  # ConvReLU(512,512):add(nn.Dropout(0.1))
            ConvReLU(512, 512),  # ConvReLU(512,512)
            nn.AvgPool2d(kernel_size=2, stride=2),  # model:add(Avg(2,2,2,2):ceil())
            #
            ConvReLU(512, 512, dropout=0.1),  # ConvReLU(512,512):add(nn.Dropout(0.1))
            ConvReLU(512, 512, dropout=0.1),  # ConvReLU(512,512):add(nn.Dropout(0.1))
            ConvReLU(512, 512),  # ConvReLU(512,512)
            nn.AvgPool2d(kernel_size=2, stride=2),  # model:add(Avg(2,2,2,2):ceil())
        )
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_ratio),  # classifier:add(nn.Dropout(0.1))
            nn.Linear(512 * 7 * 7, 4096, bias=False),  # classifier:add(nn.Linear(512,512,false))
            nn.ReLU(True),  # classifier:add(ReLU(true))
            nn.Dropout(self.dropout_ratio),  # classifier:add(nn.Dropout(0.1))
            nn.Linear(4096, 4096, bias=False),  # classifier:add(nn.Linear(512,10,false))
            nn.ReLU(True),  # classifier:add(ReLU(true))
            nn.Dropout(self.dropout_ratio),  # additional dropout
            nn.Linear(4096, num_classes, bias=False),  # additional dense layer
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                # nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # model:add(nn.View(512))
        x = self.classifier(x)
        return x


class VGG_15(nn.Module):
    def __init__(self,  dr=0.1, num_classes=1000):
        super(VGG_15, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.ReLU(),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # AvgPool2d,
            nn.ReLU(),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # AvgPool2d,
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(25088, 4096, bias=False),  # Linear,
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4096, 1000, bias=False)  # Linear,
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def vgg_15(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_15(**kwargs)
    if pretrained:
        model_path = 'model_list/vgg15_61.pth.tar'
        print('loading pre-trained model from '+model_path)
        # model_path = 'alexnet_XNOR_cpu.pth'
        pretrained_model = torch.load(model_path)
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in pretrained_model.items():
        #     name = k.replace(".module", "")  # remove `module.`
        #     new_state_dict[name] = v
        # load params

        # model.load_state_dict(pretrained_model, strict=True)
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
        # torch.save(model.state_dict(), 'vgg15_gpu.pth')
        model.load_state_dict(pretrained_model['state_dict'], strict=True)
    return model

class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''

    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        size = input.size()
        input = input.sign()
        return input

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input


class BinConv2d(nn.Module): # change the name of BinConv2d
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=-1, dropout=0.0,
            bias=True):
        super(BinConv2d, self).__init__()
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout_ratio = dropout

        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
        self.conv = nn.Conv2d(input_channels, output_channels, (3, 3), (1, 1), (1, 1), bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        binAct = BinActive.apply
        x = binAct(x)
        x = self.conv(x)
        return x

class VGG15_bn_xnor(nn.Module):
    def __init__(self,  dr=0.1, num_classes=1000):
        super(VGG15_bn_xnor, self).__init__()
        self.dr=dr
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=True),
            nn.ReLU(True),
            BinConv2d(64, 64, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BinConv2d(64, 128, kernel_size=3, stride=1),
            BinConv2d(128, 128, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BinConv2d(128, 256, kernel_size=3, stride=1),
            BinConv2d(256, 256, kernel_size=3, stride=1),
            BinConv2d(256, 256, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BinConv2d(256, 512, kernel_size=3, stride=1),
            BinConv2d(512, 512, kernel_size=3, stride=1),
            BinConv2d(512, 512, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BinConv2d(512, 512, kernel_size=3, stride=1),
            BinConv2d(512, 512, kernel_size=3, stride=1),
            BinConv2d(512, 512, kernel_size=3, stride=1),
            # nn.Dropout(dr),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        self.classifier = nn.Sequential(

            nn.Linear(512 * 7 * 7, 4096),  # Linear,
            nn.ReLU(True),
            nn.Dropout(dr),
            nn.Linear(4096, 1000)  # Linear,

        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def vgg15_bn_XNOR(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG15_bn_xnor(**kwargs)
    if pretrained:
        model_path = 'vgg15_gpu.pth'
        # model_path = 'alexnet_XNOR_cpu.pth'
        pretrained_model = torch.load(model_path)
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in pretrained_model.items():
        #     name = k.replace(".module", "")  # remove `module.`
        #     new_state_dict[name] = v
        # load params

        # model.load_state_dict(pretrained_model, strict=True)
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
        # torch.save(model.state_dict(), 'vgg15_gpu.pth')
        model.load_state_dict(pretrained_model, strict=True)
    return model

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

class VGG_15_ab(nn.Module):
    def __init__(self,  dr=0.1, num_classes=1000, linea=512*7*7):
        super(VGG_15_ab, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.AvgPool2d((2, 2), (2, 2)),
            nn.ReLU(),

            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.AvgPool2d((2, 2), (2, 2)),
            nn.ReLU(),

            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.AvgPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # AvgPool2d,
            nn.ReLU(),

            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.AvgPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # AvgPool2d,
            nn.ReLU(),

            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.AvgPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(linea, 4096, bias=False),  # Linear,
            nn.ReLU(),
            nn.Dropout(0.1),
            # nn.Linear(4096, 4096, bias=False),  # Linear,
            # nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(4096, num_classes, bias=False)  # Linear,
        )

        # self._initialize_weights()

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             # nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, 0, 0.01)
    #             # nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def vgg15ab(pretrained=None, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_15_ab(**kwargs)
    if pretrained:
        model_path = pretrained
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
