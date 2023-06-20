# python native
import os
# external library
import numpy as np
from tqdm.auto import tqdm
import albumentations as A
# torch
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import argparse
import dataset

import numpy as np
import os
import cv2
import torch
import pandas as pd
import albumentations as A


def exist_csv(filename):
    # 파일이름에 .csv 확장자 추가
    filename = filename + ".csv"

    # 중복 확인을 위해 카운터 초기화
    counter = 1

    # 파일이름이 이미 존재하는지 확인
    while os.path.exists(filename):
        # 중복되는 경우 파일이름에 숫자 추가
        filename = f"{filename[:-4]}_{counter}.csv"
        counter += 1

    return filename

def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length 
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# RLE로 인코딩된 결과를 mask map으로 복원합니다.

def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    
    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)


def test(models, data_loader, thr=1):
    models = [model.cuda().eval() for model in models]
    thr = len(models) * 0.5
    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = len(CLASSES)

        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()    
            outputs_list = []
            for model in models:
                outputs_list.append(model(images))
            outputs = torch.zeros(BATCH_SIZE, 29, 2048, 2048).cuda()
            for output in outputs_list:
                output = F.interpolate(output, size=(2048, 2048), mode="bilinear") 
                output = torch.sigmoid(output)
                outputs = outputs + output

            outputs = (outputs > thr).detach().cpu().numpy()

            for output, image_name in zip(outputs, image_names):                        #iterating batch
                for c, segm in enumerate(output):                                       #iterating classes
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
                    
    return rles, filename_and_class



def main():
    models = []
    for model in MODELS:
        models.append(torch.load(model))
    tf = A.Resize(1024, 1024)

    test_dataset = dataset.XRayInferenceDataset(transforms=tf)          #1024 모델을 위한 데이터 로더
                                                                        #다른 input의 모델은 추가적인 데이터 로더를 구성해서 넣어주어야함
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    rles, filename_and_class = test(models, test_loader)

    classes, filename = zip(*[x.split("_") for x in filename_and_class])

    image_name = [os.path.basename(f) for f in filename]

    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })

    # save_path = f"{SAVED_DIR}/{'_'.join(RESULTPATH)}"     #GPU 남으면 테스트
    save_path = f"{SAVED_DIR}/softvoting"

    csv_name = exist_csv(save_path)
    df.to_csv(csv_name, index=False)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=2, help='input batch size for validing (default: 2)')
    parser.add_argument('--data_path', type=str, default='/opt/ml/input/data/test/DCM')
    parser.add_argument('--output_path', type=str, default='/opt/ml/input/result')
    parser.add_argument('--models', nargs='+', help='Input a list')
    
    args = parser.parse_args()
    
    BATCH_SIZE = args.batch_size
    IMAGE_ROOT = args.data_path
    SAVED_DIR = args.output_path
    MODELS = args.models
    if not os.path.isdir(SAVED_DIR):                                                           
        os.mkdir(SAVED_DIR)
    
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
    
    # RESULTPATH = [model.split('/')[-1].split('.')[0] for model in args.models]
    # print(RESULTPATH)
    # print(models)

    #경로 설정 완료 되면 아래 줄 제거
    # MODEL_NAME = MODEL_ROOT.split('/')[-1].split('.')[0] # 절대 경로 제거
    # MODEL_NAME = MODEL_NAME.replace('_best_model', '')

    # MODEL_NAME2 = MODEL_ROOT2.split('/')[-1].split('.')[0] # 절대 경로 제거
    # MODEL_NAME2 = MODEL_NAME2.replace('_best_model', '')
    
    main()