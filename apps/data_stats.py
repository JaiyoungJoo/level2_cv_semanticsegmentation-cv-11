import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
# import seaboern as sns
import os
import json
from dataset import XRayDataset, XRayDataset_path
from torch.utils.data import DataLoader, Subset
import torch
import albumentations as A
import random
from tqdm.auto import tqdm
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import time
import cv2
import tempfile
from torchvision.transforms.functional import resize

MODEL_PATH = "/opt/ml/input/weights/final/Pretrained_smp_densenet161_comb_loss_tf=True_cln=True_e=100_sd=up.pt"

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

BATCH_SIZE=8

# define colors
PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]

# utility function
# this does not care overlap
def label2rgb(label):
    if isinstance(label, torch.Tensor):
        label_cpu=label.cpu()
        label=label_cpu.numpy()
    image_size = label.shape[1:] + (3, )
    image = np.zeros(image_size, dtype=np.uint8)
    
    for i, class_label in enumerate(label):
        image[class_label == 1] = PALETTE[i]
        
    return image

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    
    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
# def make_dataset(debug="False"):
#     tf = A.Resize(512, 512)
#     valid_dataset = XRayDataset(is_train=False, transforms=tf)
#     valid_loader = DataLoader(
#         dataset=valid_dataset, 
#         batch_size=2,
#         shuffle=False,
#         num_workers=0,
#         drop_last=False
#     )
#     return valid_loader

def convertimg(image, annotations):
    image = image / 255.
    label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
    label = np.zeros(label_shape, dtype=np.uint8)
    
    # read label file
    annotations = annotations["annotations"]
    
    # iterate each class
    for ann in annotations:
        c = ann["label"]
        class_ind = CLASS2IND[c]
        points = np.array(ann["points"])
        
        # polygon to mask
        class_label = np.zeros([2048,2048], dtype=np.uint8)
        cv2.fillPoly(class_label, [points], 1)
        label[..., class_ind] = class_label

    # to tenser will be done later
    image = image.transpose(2, 0, 1)    # make channel first
    label = label.transpose(2, 0, 1)
    
    image = torch.from_numpy(image).float()
    label = torch.from_numpy(label).float()
        
    return image, label

def validation(epoch, model, image, mask, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.eval()

    dices = []
    all_dices = []
    all_masks = []
    with torch.no_grad():
        total_loss = 0
        cnt = 0
        image, mask = convertimg(image,mask)
        image = resize(image,(1024,1024))
        image = image.unsqueeze(0).cuda()
        mask = mask.unsqueeze(0).cuda()
        
        outputs = model(image)
        
        output_h, output_w = outputs.size(-2), outputs.size(-1)
        mask_h, mask_w = mask.size(-2), mask.size(-1)
            
        # restore original size
        if output_h != mask_h or output_w != mask_w:
            outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
        
        outputs = torch.sigmoid(outputs)
        # outputs = (outputs > thr)
        # np.array(images) # Original image
        # np.array(masks) # GT
        # np.array(outputs) # Model inference
        dice = dice_coef(outputs, mask)  # dice
        all_dices.append(dice.mean(axis=1)) 
        tp = torch.logical_and(outputs, mask).detach().cpu().squeeze()
        fp = torch.logical_and(torch.logical_not(outputs > thr), mask).detach().cpu().squeeze()
        fn = torch.logical_and(torch.logical_not(mask), outputs > thr).detach().cpu().squeeze()
        dices.append(dice) #
    dices = torch.cat(dices, 0)
    return [mask, tp, fp, fn, all_dices]


def app():
    st.title("Error Analysis")
    st.markdown("---")
    upload_img = st.file_uploader('이미지를 업로드 하세요.', type=['png', 'jpg', 'jpeg'])
    upload_json = st.file_uploader('Json 파일을 업로드 하세요.', type=['json'])
    if upload_img is not None and upload_json is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(upload_img.read())
        image = cv2.imread(temp_file.name)
        mask = json.load(upload_json)
        model = torch.load(MODEL_PATH)
        set_seed(21)
        gtmask, tp, fp, fn, dices = validation(1, model, image, mask)

        # masks = np.concatenate(masks, axis=0)   #predicetd mask
        # dices = np.concatenate(dices.tolist(), axis=0)   #predicted labels' dices
        # gtimage = np.concatenate(gtimage, axis=0)   #GT image(2048, 2048)
        # gtmask = np.concatenate(gtmask.tolist(), axis=0)    #GT mask(2048, 2048)
        # tp=np.concatenate(tp.tolist(), axis=0)
        # fn=np.concatenate(fn.tolist(), axis=0)
        # fp=np.concatenate(fp.tolist(), axis=0)
        # st.write(type(gtmask[0]))

        with st.container():
            st.subheader("Dice: "+str(dices[0].item()))
            fig, ax = plt.subplots(1, 4)
            ax = ax.flatten()
            # fig.suptitle("Dice: " + str(dices[0]), fontsize=10)
            ax[0].imshow(image)   # remove channel dimension
            ax[1].imshow(image)   # remove channel dimension
            ax[2].imshow(image)   # remove channel dimension
            ax[3].imshow(image)   # remove channel dimension
            ax[0].axis("off")
            ax[1].axis("off")
            ax[2].axis("off")
            ax[3].axis("off")
            ax[0].set_title("GT")
            ax[1].set_title("TP")
            ax[2].set_title("FN")
            ax[3].set_title("FP")
        
        
            ax[0].imshow(label2rgb(gtmask[0]), alpha=0.5)
            ax[1].imshow(label2rgb(tp), alpha=0.5)
            ax[2].imshow(label2rgb(fn), alpha=0.7)
            ax[3].imshow(label2rgb(fp), alpha=0.7)
            st.pyplot(fig)
            st.markdown(
            """
            <style>
            .css-1v0mbdj{
                    border: 3px solid #e5f0f9;
                    background-color: #e5f0f9;
                    border-radius: 10px;
                    box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.5);
                    padding: 15px
                }
            </style>
            """,
            unsafe_allow_html=True,
        )
        
    
