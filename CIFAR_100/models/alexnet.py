import torch
import os
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

__all__ = ['AlexNet', 'alexnet']


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


class BinConv2d(nn.Module):  # change the name of BinConv2d
    def __init__(self, input_channels, output_channels,
                 kernel_size=-1, stride=-1, padding=-1, groups=1, dropout=0.0,
                 Linear=False):
        super(BinConv2d, self).__init__()
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout

        if dropout != 0:
            self.dropout = nn.Dropout(dropout)
        self.Linear = Linear
        if not self.Linear:
            self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.conv = nn.Conv2d(input_channels, output_channels,
                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        else:
            self.bn = nn.BatchNorm1d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.linear = nn.Linear(input_channels, output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        binAct = BinActive.apply
        x = binAct(x)
        if self.dropout_ratio != 0:
            x = self.dropout(x)
        if not self.Linear:
            x = self.conv(x)
        else:
            x = self.linear(x)
        x = self.relu(x)
        return x


class AlexNet_XNOR(nn.Module):
    def __init__(self, num_classes=100):
        super(AlexNet_XNOR, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BinConv2d(64, 192, kernel_size=3, stride=1, padding=1, groups=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BinConv2d(192, 384, kernel_size=3, stride=1, padding=1),
            BinConv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=1),
            BinConv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            BinConv2d(1024, 4096, Linear=True, dropout=0.5),
            BinConv2d(4096, 4096, Linear=True, dropout=0.5),
            nn.BatchNorm1d(4096, eps=1e-3, momentum=0.1, affine=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 1024)
        x = self.classifier(x)
        return x

class AlexNet_BWN_flatten(nn.Module):
    def __init__(self, num_classes=100):
        super(AlexNet_BWN_flatten, self).__init__()
        self.num_classes = num_classes
        self.dr = 0.1
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(96, eps=1e-4, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dr),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dr),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 4096, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dr),
            nn.Linear(4096, 4096, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes, bias=False),
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
        x = x.view(x.size(0), 1024)
        x = self.classifier(x)
        return x

def alexnet_xnor(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet_XNOR(**kwargs)
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

def alexnet_bwn(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet_BWN_flatten(**kwargs)
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