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
    def backward(self, grad_output): #grad_output is the estimated weight
        # loss = T.mean(T.sqr(T.maximum(0., 1. - target * train_output)))
        # print('output')
        # print(grad_output.shape)
        # print('output_ened')
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        # print('input')
        # print(grad_input.shape)
        # print('input_ened')
        return grad_input

# class BinConv2d(nn.Module): # change the name of BinConv2d
#     def __init__(self, input_channels, output_channels,
#             kernel_size=-1, stride=-1, padding=-1, groups=1, dropout=0.0,
#             Linear=False):
#         super(BinConv2d, self).__init__()
#         self.layer_type = 'BinConv2d'
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dropout_ratio = dropout
#
#         if dropout!=0:
#             self.dropout = nn.Dropout(dropout)
#         self.Linear = Linear
#         if not self.Linear:
#             # self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
#             self.conv = nn.Conv2d(input_channels, output_channels,
#                     kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
#         else:
#             # self.bn = nn.BatchNorm1d(input_channels, eps=1e-4, momentum=0.1, affine=True)
#             self.linear = nn.Linear(input_channels, output_channels)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         # x = self.bn(x)
#         # binAct = BinActive.apply
#         # x = binAct(x)
#         # x=BinActive()(x)
#         if self.dropout_ratio!=0:
#             x = self.dropout(x)
#         if not self.Linear:
#             x = self.conv(x)
#         else:
#             x = self.linear(x)
#         x = self.relu(x)
#         return x


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


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            BinConv2d(64, 192, kernel_size=3, stride=1, padding=1, groups=1),
            nn.MaxPool2d(kernel_size=2),
            BinConv2d(192, 384, kernel_size=3, stride=1, padding=1),
            BinConv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=1),
            BinConv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=1),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            BinConv2d(256 * 2 * 2, 4096, Linear=True),
            BinConv2d(4096, 4096, dropout=0.5, Linear=True),
            nn.BatchNorm1d(4096, eps=1e-3, momentum=0.1, affine=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x
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


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model_path = 'model_list/alexnet.pth.tar'
        pretrained_model = torch.load(model_path)
        model.load_state_dict(pretrained_model['state_dict'], strict=False)
    return model
