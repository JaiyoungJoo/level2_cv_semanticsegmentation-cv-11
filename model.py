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
    

# multi modal
class MultiModal(nn.Module):
    def __init__(self, in_features=64, encoder = 'resnet101'):
        super().__init__()

        self.backbone = FCN(num_classes= in_features ,encoder=encoder)

        self.age_fc = nn.Linear(1, in_features)
        self.gender_fc = nn.Linear(1, in_features)
        self.wieght_fc = nn.Linear(1, in_features)
        self.hight_fc = nn.Linear(1, in_features)

        self.branch_gender = nn.Linear(in_features=in_features*512*512, out_features=1)
        self.branch_age = nn.Linear(in_features=in_features*512*512, out_features=1)
        self.branch_weight = nn.Linear(in_features=in_features*512*512, out_features=1)
        self.branch_hight = nn.Linear(in_features=in_features*512*512, out_features=1)


        self.decoder = nn.Sequential(
            # Define the decoder layers, such as upsampling and transposed convolutions
            nn.ConvTranspose2d(in_features*5, in_features*2, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features*2, in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, 29, kernel_size=1, stride=1),
        )
    def forward(self, x, age, gender, weight, hight):
        out = self.backbone(x)
        age = self.age_fc(age)
        gender = self.gender_fc(gender)
        weight = self.weight_fc(weight)
        hight = self.hight_fc(hight)
        out_all = torch.cat([out, age.unsqueeze(2).unsqueeze(3).repeat(1, 1, 512, 512)], dim=1)
        out_all = torch.cat([out_all, gender.unsqueeze(2).unsqueeze(3).repeat(1, 1, 512, 512)], dim=1)
        out_all = torch.cat([out_all, weight.unsqueeze(2).unsqueeze(3).repeat(1, 1, 512, 512)], dim=1)
        out_all = torch.cat([out_all, hight.unsqueeze(2).unsqueeze(3).repeat(1, 1, 512, 512)], dim=1)

        out_segment = self.decoder(out_all)
        x = out.view(out.size(0),-1)
        out_gender = self.branch_gender(x)
        out_age = self.branch_age(x)
        out_weight = self.branch_weight(x)
        out_hight = self.branch_hight(x)

        
        return  out_segment, out_gender, out_age, out_weight, out_hight