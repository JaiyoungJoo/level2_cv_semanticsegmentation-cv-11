# torch
import torch.nn as nn
from torchvision.models import segmentation
import segmentation_models_pytorch as smp
import torch
import hrnet
import torch.nn.functional as F


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
    
class FPN_gray(nn.Module):
    def __init__(self, num_classes=29, encoder = "resnet50"):
        super().__init__()
        self.net = smp.FPN(
            encoder_name=encoder,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                     # model output channels (number of classes in your dataset)
            activation="identity"
        )

    def forward(self, x):
        out = self.net(x)
        return out
    
class MAnet(nn.Module):
    def __init__(self, num_classes=29, encoder = "resnet50"):
        super().__init__()
        self.net = smp.MAnet(
            encoder_name=encoder,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=num_classes,                     # model output channels (number of classes in your dataset)
            activation="identity"
        )

    def forward(self, x):
        out = self.net(x)
        return out
      
# hrnet
class  HRNet(nn.Module):
    def __init__(self, name='hrnet48',pretrained ="/opt/ml/weights/MultiModalV2/hrnetv2_w48_imagenet_pretrained.pth",num_classes=29, encoder = 'HR'):
        super().__init__()
        self.net = hrnet.get_ocr_model(name = name, pretrained=pretrained)
    
    def forward(self, x):
        out = self.net(x)
        out = F.interpolate(out, size=(512, 512), mode="bilinear")
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
    
class Pretrained_Multimodal(nn.Module):
    def __init__(self,  model = "/opt/ml/weights/fpn_resnet101_best_model.pt"):
        super().__init__()
        self.net = torch.load(model)

    def forward(self, x, age, gender, weight, hight):
        out_segment, out_age, out_gender, out_weight, out_hight = self.net(x, age, gender, weight, hight)
        return out_segment, out_age, out_gender, out_weight, out_hight
    

# multi modal
class MultiModal(nn.Module):
    def __init__(self, in_features=30, encoder = 'densenet169'):
        super().__init__()

        self.backbone = FPN(num_classes= in_features ,encoder=encoder)

        self.age_fc = nn.Linear(1, in_features)
        self.gender_fc = nn.Linear(1, in_features)
        self.weight_fc = nn.Linear(1, in_features)
        self.hight_fc = nn.Linear(1, in_features)
        
        self.downsampling = nn.Conv2d(in_features, in_features, kernel_size=5, stride=5, padding=1)

        self.branch_age = nn.Sequential(
                nn.Linear(in_features=312120, out_features=51),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(51,1),
                nn.Softmax()
            )
        self.branch_gender = nn.Sequential(
                nn.Linear(in_features=312120, out_features=52),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(52,1),
                nn.Softmax()
            )
        self.branch_weight = nn.Sequential(
                nn.Linear(in_features=312120, out_features=51),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(51,1),
                nn.Softmax()
            )
        self.branch_hight = nn.Sequential(
                nn.Linear(in_features=312120, out_features=51),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(51,1),
                nn.Softmax()
            )


        self.decoder = nn.Sequential(
            # Define the decoder layers, such as upsampling and transposed convolutions
            nn.ConvTranspose2d(in_features*5, in_features*2, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features*2, in_features, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, 29, kernel_size=1, stride=1),
        )
    def forward(self, x, age, gender, weight, hight):
        out = self.backbone(x)
        age = self.age_fc(age.unsqueeze(1))

        gender = self.gender_fc(gender.unsqueeze(1))
        weight = self.weight_fc(weight.unsqueeze(1))
        hight = self.hight_fc(hight.unsqueeze(1))
        out_all = torch.cat([out, age.unsqueeze(2).unsqueeze(3).repeat(1, 1, 512, 512)], dim=1)
        out_all = torch.cat([out_all, gender.unsqueeze(2).unsqueeze(3).repeat(1, 1, 512, 512)], dim=1)
        out_all = torch.cat([out_all, weight.unsqueeze(2).unsqueeze(3).repeat(1, 1, 512, 512)], dim=1)
        out_all = torch.cat([out_all, hight.unsqueeze(2).unsqueeze(3).repeat(1, 1, 512, 512)], dim=1)

        out_segment = self.decoder(out_all)
        out_down = self.downsampling(out)
        out_flatten = out_down.view(out_down.size(0),-1)
        
        out_age = self.branch_age(out_flatten)
        out_gender = self.branch_gender(out_flatten)
        out_weight = self.branch_weight(out_flatten)
        out_hight = self.branch_hight(out_flatten)

        
        return  out_segment, out_age, out_gender, out_weight, out_hight
    
class MultiModalV2(nn.Module):
    def __init__(self, in_features=29, encoder = 'densenet169'):
        super().__init__()

        self.backbone = hrnet.get_ocr_model(name='hrnet48',pretrained ="/opt/ml/weights/hrnetv2_w48_imagenet_pretrained.pth")

        self.age_fc = nn.Linear(1, in_features)
        self.gender_fc = nn.Linear(2, in_features)
        self.weight_fc = nn.Linear(1, in_features)
        self.hight_fc = nn.Linear(1, in_features)
        
        # self.downsampling = nn.Conv2d(in_features, in_features, kernel_size=5, stride=5, padding=1)

        self.branch_age = nn.Sequential(
                nn.Conv2d(in_features, in_features, kernel_size=5, stride=5, padding=1),
                nn.Flatten(),
                nn.Linear(in_features=19604, out_features=51),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(51,1),
                nn.Softmax()
            )
        self.branch_gender = nn.Sequential(
                nn.Conv2d(in_features, in_features, kernel_size=5, stride=5, padding=1),
                nn.Flatten(),
                nn.Linear(in_features=19604, out_features=52),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(52,2),
                nn.Softmax()
            )
        self.branch_weight = nn.Sequential(
                nn.Conv2d(in_features, in_features, kernel_size=5, stride=5, padding=1),
                nn.Flatten(),
                nn.Linear(in_features=19604, out_features=51),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(51,1),
                nn.Softmax()
            )
        self.branch_hight = nn.Sequential(
                nn.Conv2d(in_features, in_features, kernel_size=5, stride=5, padding=1),
                nn.Flatten(),
                nn.Linear(in_features=19604, out_features=51),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(51,1),
                nn.Softmax()
            )


        self.decoder = nn.Sequential(
            # Define the decoder layers, such as upsampling and transposed convolutions
            nn.ConvTranspose2d(in_features*5, in_features*3, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_features*3, in_features*2, kernel_size=2, stride=2),
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

        out_all = torch.cat([out, age.unsqueeze(2).unsqueeze(3).repeat(1, 1, 128, 128)], dim=1)
        out_all = torch.cat([out_all, gender.unsqueeze(2).unsqueeze(3).repeat(1, 1, 128, 128)], dim=1)
        out_all = torch.cat([out_all, weight.unsqueeze(2).unsqueeze(3).repeat(1, 1, 128, 128)], dim=1)
        out_all = torch.cat([out_all, hight.unsqueeze(2).unsqueeze(3).repeat(1, 1, 128, 128)], dim=1)

        out_segment = self.decoder(out_all)
        #out_down = self.downsampling(out)
        #out_flatten = out_down.view(out_down.size(0),-1)
        
        out_age = self.branch_age(out) # out_flatten -> out
        out_gender = self.branch_gender(out) # out_flatten -> out
        out_weight = self.branch_weight(out) # out_flatten -> out
        out_hight = self.branch_hight(out) # out_flatten -> out
        
        return  out_segment, out_age, out_gender, out_weight, out_hight