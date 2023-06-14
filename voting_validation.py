# python native
# external library
from tqdm.auto import tqdm
import albumentations as A
# torch
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import argparse
import dataset

import torch
import albumentations as A

BATCH_SIZE = 2
CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    
    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

def test(model, model2, data_loader, thr=1):
    model = model.cuda()
    model.eval()
    dices = []

    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = len(CLASSES)

        for step, (images, image_names, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()    
            outputs = model(images)
            outputs2 = model2(images)
            masks = masks.detach().cpu()
            
            # restore original size
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")        #(batch, 29, 2048, 2048)
            outputs = torch.sigmoid(outputs)

            
            # restore original size
            outputs2 = F.interpolate(outputs2, size=(2048, 2048), mode="bilinear")        #(batch, 29, 2048, 2048)
            outputs2 = torch.sigmoid(outputs2)

            outputs = outputs + outputs2
            outputs = (outputs > thr).detach().cpu()
            
            dice = dice_coef(outputs, masks)
            dices.append(dice)
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)

    return dices_per_class, filename_and_class




def main():
    model = torch.load(MODEL_ROOT)
    model2 = torch.load(MODEL_ROOT2)
    

    tf = A.Resize(512, 512)

    test_dataset  = dataset.XRayDataset_valid(is_train = False, transforms=tf)

    test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    drop_last=False
    )

    rles, filename_and_class = test(model, model2, test_loader)

    for i in range(len(rles)):
        print(f"{CLASSES[i]}:  {rles[i]}")

    print(f"dice mean: {rles.mean()}")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=2, help='input batch size for validing (default: 2)')
    # Container environment
    parser.add_argument('--model_path', type=str, default='/opt/ml/input/result/deeplabv3_resnet101_best_model_seedup.pt')
    parser.add_argument('--model_path2', type=str, default='/opt/ml/input/result/deeplabv3_resnet101_best_model_seedup.pt')
    
    args = parser.parse_args()
    
    BATCH_SIZE = args.batch_size
    MODEL_ROOT = args.model_path
    MODEL_ROOT2 = args.model_path2
    CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',]

    # mask map으로 나오는 인퍼런스 결과를 RLE로 인코딩 합니다.
    CLASS2IND = {v: i for i, v in enumerate(CLASSES)}

    IND2CLASS = {v: k for k, v in CLASS2IND.items()}
    
    MODEL_NAME = MODEL_ROOT.split('/')[-1].split('.')[0] # 절대 경로 제거
    MODEL_NAME = MODEL_NAME.replace('_best_model', '')

    MODEL_NAME2 = MODEL_ROOT2.split('/')[-1].split('.')[0] # 절대 경로 제거
    MODEL_NAME2 = MODEL_NAME2.replace('_best_model', '')
    
    main()