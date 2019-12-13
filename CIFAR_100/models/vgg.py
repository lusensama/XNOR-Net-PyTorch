import torch
import torch.nn as nn

class VGG_15_max(nn.Module):
    def __init__(self,  dr=0.1, num_classes=100):
        super(VGG_15_max, self).__init__()
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
            nn.Linear(512, 4096, bias=False),  # Linear,
            nn.ReLU(),
            nn.Dropout(0.1),
            # nn.Linear(4096, 4096, bias=False),  # Linear,
            # nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(4096, num_classes, bias=False)  # Linear,
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
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def vgg_15_max(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_15_max(**kwargs)
    if pretrained:
        model_path = 'vgg15m_cifar100model_best.pth.tar'
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

class VGG_15_avg(nn.Module):
    def __init__(self,  dr=0.1, num_classes=1000, linea=512*7*7):
        super(VGG_15_avg, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),

            nn.ReLU(),
            nn.AvgPool2d((2, 2), (2, 2)),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),

            nn.ReLU(),
            nn.AvgPool2d((2, 2), (2, 2)),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),

            nn.ReLU(),
            nn.AvgPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # AvgPool2d,
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),

            nn.ReLU(),
            nn.AvgPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # AvgPool2d,
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),

            nn.ReLU(),
            nn.AvgPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)
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
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def vgg_15_avg(pretrained=False, dataset='cifar100' , **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if dataset == 'imagenet':
        model = VGG_15_avg(num_classes=1000, **kwargs)
    elif dataset == 'cifar100':
        model = VGG_15_avg(num_classes=100, linea=512,**kwargs)
    if pretrained:
        model_path = 'vgg15avg.pth.tar'
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
class VGG_15_avg2(nn.Module):
    def __init__(self,  dr=0.1, num_classes=1000, linea=512*7*7):
        super(VGG_15_avg2, self).__init__()
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
            nn.AvgPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # AvgPool2d,
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
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def vgg_15_avg2(pretrained=False, dataset='cifar100' , **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if dataset == 'imagenet':
        model = VGG_15_avg2(num_classes=1000, **kwargs)
    elif dataset == 'cifar100':
        model = VGG_15_avg2(num_classes=100, linea=512,**kwargs)
    if pretrained:
        model_path = 'vgg15avg.pth.tar'
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

class VGG_15_XNOR(nn.Module):
    def __init__(self,  dr=0.1, num_classes=100):
        super(VGG_15_XNOR, self).__init__()
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
            # nn.AdaptiveAvgPool2d((7, 7))
        )
        self.classifier = nn.Sequential(

            nn.Linear(512, 4096),  # Linear,
            nn.ReLU(True),
            nn.Dropout(dr),
            nn.Linear(4096, num_classes)  # Linear,

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
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def vgg_15_xnor(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_15_XNOR(**kwargs)
    if pretrained:
        model_path = 'models/vgg15_xnor_cifar100.pth.tar'
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