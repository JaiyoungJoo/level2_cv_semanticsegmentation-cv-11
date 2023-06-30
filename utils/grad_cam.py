import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch
import torch.functional as F
import numpy as np
import requests
import torchvision
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import cv2
import argparse
from torchvision import models
from pytorch_grad_cam import GradCAM
import os

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', default='/opt/ml/input/data/train/DCM/')
parser.add_argument('--pt_path', default="/opt/ml/input/code/results_baseline/fcn_resnet101_best_model81.pt")
args = parser.parse_args()
classname='Trapezium'
validimg=['ID004/image1661144691792.png', 'ID004/image1661144724044.png', 'ID009/image1661145407161.png', 'ID009/image1661145432967.png', 'ID014/image1661216876670.png', 'ID014/image1661216907060.png', 'ID019/image1661219523078.png', 'ID019/image1661219796151.png', 'ID024/image1661304293036.png', 'ID024/image1661304319731.png', 'ID029/image1661306136685.png', 'ID029/image1661306162532.png', 'ID034/image1661317748865.png', 'ID034/image1661317775801.png', 'ID039/image1661318938038.png', 'ID039/image1661318964936.png', 'ID054/image1661391074265.png', 'ID054/image1661391105683.png', 'ID060/image1661393384829.png', 'ID060/image1661393400879.png', 'ID064/image1661734980185.png', 'ID064/image1661735006875.png', 'ID070/image1661736017073.png', 'ID070/image1661736042863.png', 'ID075/image1661736870231.png', 'ID075/image1661736898823.png', 'ID080/image1661737362347.png', 'ID080/image1661737404974.png', 'ID085/image1661737831600.png', 'ID085/image1661737858331.png', 'ID090/image1661821775105.png', 'ID090/image1661821802814.png', 'ID095/image1661822623891.png', 'ID095/image1661822638602.png', 'ID100/image1661823638297.png', 'ID100/image1661823665932.png', 'ID105/image1661824735289.png', 'ID105/image1661824761643.png', 'ID111/image1661907834003.png', 'ID111/image1661907848327.png', 'ID115/image1661908159643.png', 'ID115/image1661908185790.png', 'ID120/image1661908761598.png', 'ID120/image1661908778667.png', 'ID125/image1661910199152.png', 'ID125/image1661910238659.png', 'ID130/image1662339673237.png', 'ID130/image1662339722889.png', 'ID136/image1662340433669.png', 'ID136/image1662340460163.png', 'ID278/image1664155311840.png', 'ID278/image1664155340191.png', 'ID283/image1664155969209.png', 'ID283/image1664155987422.png', 'ID288/image1664156956152.png', 'ID288/image1664156981600.png', 'ID293/image1664157316352.png', 'ID293/image1664157343885.png', 'ID298/image1664240709270.png', 'ID298/image1664240724225.png', 'ID303/image1664241119204.png', 'ID303/image1664241147172.png', 'ID308/image1664241429399.png', 'ID308/image1664241445936.png', 'ID313/image1664241736840.png', 'ID313/image1664241753946.png', 'ID318/image1664242930720.png', 'ID318/image1664242958139.png', 'ID323/image1664846188621.png', 'ID323/image1664846204441.png', 'ID328/image1664846838406.png', 'ID328/image1664846856735.png', 'ID333/image1664847413470.png', 'ID333/image1664847440074.png', 'ID338/image1664848589440.png', 'ID338/image1664848616528.png', 'ID343/image1664932424514.png', 'ID343/image1664932451633.png', 'ID348/image1664933179818.png', 'ID348/image1664933195576.png', 'ID353/image1664934044092.png', 'ID353/image1664934060702.png', 'ID358/image1664934876849.png', 'ID358/image1664934894024.png', 'ID363/image1664935962797.png', 'ID363/image1664935989808.png', 'ID368/image1665450162776.png', 'ID368/image1665450178483.png', 'ID373/image1665452035077.png', 'ID373/image1665452058937.png', 'ID378/image1665452927615.png', 'ID378/image1665452941530.png', 'ID383/image1665454683455.png', 'ID383/image1665454711973.png', 'ID388/image1665536805845.png', 'ID388/image1665536821260.png', 'ID393/image1665537862361.png', 'ID393/image1665537890819.png', 'ID399/image1665539161576.png', 'ID399/image1665539181057.png', 'ID403/image1665539937219.png', 'ID403/image1665539962902.png', 'ID408/image1665540854846.png', 'ID408/image1665540873549.png', 'ID413/image1666054988891.png', 'ID413/image1666055014770.png', 'ID418/image1666055793379.png', 'ID418/image1666055808396.png', 'ID423/image1666058234645.png', 'ID423/image1666058249428.png', 'ID428/image1666059689193.png', 'ID428/image1666059716951.png', 'ID433/image1666060113452.png', 'ID433/image1666060142188.png', 'ID438/image1666141346033.png', 'ID438/image1666141363652.png', 'ID443/image1666144094464.png', 'ID443/image1666144111171.png', 'ID448/image1666573448004.png', 'ID448/image1666573462779.png', 'ID453/image1666573743950.png', 'ID453/image1666573768043.png', 'ID458/image1666575044555.png', 'ID458/image1666575070798.png', 'ID463/image1666575511336.png', 'ID463/image1666575527139.png', 'ID468/image1666659863512.png', 'ID468/image1666659890125.png', 'ID473/image1666660367473.png', 'ID473/image1666660395613.png', 'ID478/image1666660861462.png', 'ID478/image1666660904025.png', 'ID483/image1666661332742.png', 'ID483/image1666661359745.png', 'ID488/image1666662049185.png', 'ID488/image1666662075807.png', 'ID493/image1666662711461.png', 'ID493/image1666662727834.png', 'ID499/image1666746427808.png', 'ID499/image1666746454260.png', 'ID503/image1666746789169.png', 'ID503/image1666746807797.png', 'ID508/image1666747732146.png', 'ID508/image1666747749143.png', 'ID513/image1666748180196.png', 'ID513/image1666748198922.png', 'ID518/image1666749056245.png', 'ID518/image1666749083578.png', 'ID523/image1667178735444.png', 'ID523/image1667178762956.png', 'ID528/image1667180274249.png', 'ID528/image1667180301380.png', 'ID533/image1667265171128.png', 'ID533/image1667265191721.png', 'ID538/image1667266190612.png', 'ID538/image1667266231591.png', 'ID543/image1667266674012.png', 'ID543/image1667266700981.png', 'ID548/image1667354140846.png', 'ID548/image1667354167046.png']


class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model): 
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model
        
    def forward(self, x):
        return self.model(x)["out"]
class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        return (model_output[self.category, :, : ] * self.mask).sum()



def grad_cam(path,classname):
    image = np.array(Image.open(os.path.join(args.img_path,path)))
    image = np.stack([image] * 3, axis=-1)
    image = cv2.resize(image, None, fx=0.5, fy=0.5)
    
    rgb_img = np.float32(image) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    # Taken from the torchvision tutorial
    # https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html
    # model = deeplabv3_resnet50(pretrained=True, progress=False)
    # model = torch.load("/opt/ml/input/code/results_baseline/fcn_resnet50_best_model.pt")
    # model = torch.load(args.pt_path)
    model = torch.load(args.pt_path)
    print(model)
    model = model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
        input_tensor = input_tensor.cuda()
    
    output = model(input_tensor)
    print(type(output), output.keys())
    
    model = SegmentationModelOutputWrapper(model)
    output = model(input_tensor)

    normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()
    sem_classes = [
        'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
        'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
        'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
        'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
        'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
        'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
    ]
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
    xray_category = sem_class_to_idx[classname]
    xray_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
    xray_mask_uint8 = 255 * np.uint8(xray_mask == xray_category)
    xray_mask_float = np.float32(xray_mask == xray_category)
    
    target_layers = [model.model.backbone.layer4]
    targets = [SemanticSegmentationTarget(xray_category, xray_mask_float)]
    with GradCAM(model=model,
                 target_layers=target_layers,
                 use_cuda=torch.cuda.is_available()) as cam:
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets)[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        # print('cam_image.type', type(cam_image))
        tosaveimg = Image.fromarray(cam_image)
        # print('2')
        fname=path.split('/')[1]
        tosaveimg.save(f'/opt/ml/input/level2_cv_semanticsegmentation-cv-11/grad_images/{classname}/{fname}')
        # cv2.imwrite('/opt/ml/input/code/grad_images/sample_fcn.png',cam_image)
        print('img SAVE DONE!')


for i in validimg:
    grad_cam(i,classname)
    
