# torch
import torch.nn as nn
from torchvision.models import segmentation
from




class deeplabv3_resnet50_(nn.Module):
    def __init__(self, num_classes=128):
        super().__init__()
        self.net = segmentation.deeplabv3_resnet50(pretrained=True)
        self.net.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        out = self.net(x)['out']
        age = self.age_fc(age)
        gender = self.gender_fc(gender)
       
        return out, age, gender