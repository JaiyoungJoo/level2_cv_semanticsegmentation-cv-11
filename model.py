# torch
import torch.nn as nn
from torchvision.models import segmentation
import segmentation_models_pytorch as smp
import torch



# torchvision
class Deeplabv3(nn.Module):
    def __init__(self, num_classes=29, encoder = "resnet50"):
        super().__init__()
        if encoder == "resnet50":
            self.net = segmentation.deeplabv3_resnet50(pretrained=True)
        elif encoder == "resnet101":
            self.net = segmentation.deeplabv3_resnet101(pretrained=True)
            
        self.net.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        out = self.net(x)['out']
        return out
    
class FCN(nn.Module):
    def __init__(self, num_classes=29, encoder = "resnet50"):
        super().__init__()
        if encoder == "resnet50":
            self.net = segmentation.fcn_resnet50(pretrained=True)
        elif encoder == "resnet101":
            self.net = segmentation.fcn_resnet101(pretrained=True)
        self.net.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        out = self.net(x)['out']
        return out
# SMP

class FPN(nn.Module):
    def __init__(self, num_classes=29, encoder = "resnet50"):
        super().__init__()
        self.net = smp.FPN(
            encoder_name=encoder,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=num_classes,                     # model output channels (number of classes in your dataset)
            activation="identity"
        )

    def forward(self, x):
        out = self.net(x)
        return out
    
# pretrained weight

class Pretrained_torchvision(nn.Module):
    def __init__(self,  model = "/opt/ml/weights/fcn_resnet101_best_model.pt"):
        super().__init__()
        self.net = torch.load(model)

    def forward(self, x):
        out = self.net(x)['out']
        return out
    
class Pretrained_smp(nn.Module):
    def __init__(self,  model = "/opt/ml/weights/fpn_resnet101_best_model.pt"):
        super().__init__()
        self.net = torch.load(model)

    def forward(self, x):
        out = self.net(x)
        return out