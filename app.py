import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import time
import cv2
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from inference import encode_mask_to_rle
from visualizatioin import label2rgb, decode_rle_to_mask, all_label2rgb
import inference
import albumentations as A
from torch.utils.data import Dataset, DataLoader
import dataset

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

# @st.cache_data
# def run_model(_model, _inputs):
#     # model = load_model()
#     return _model(_inputs)

@st.cache_resource
def load_model():
    model = torch.load('/opt/ml/input/weights/FPN_densenet161_150/FCN_resnet50_bce_loss_200_dataclean3_noacc_127e.pt')
    model.eval()
    return model

@st.cache(allow_output_mutation=True)
def infer(outputs):
    return outputs

def test(model, data_loader, thr=0.5):
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = len(CLASSES)

        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()

            # outputs = model(images)['out']
            outputs = model(images)
 
            # restore original size
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()
   
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
   
    return rles, filename_and_class

def main():

    st.set_page_config(layout="wide")
    st.title(":hand: Hand :bone: Bone :blue[Segmentation]")
    st.sidebar.title('Select part of bone')
    # st.sidebar.checkbox('체크박스에 표시될 문구')
    upload_file = st.file_uploader('이미지를 업로드 하세요.', type=['png', 'jpg', 'jpeg'])

    checkbox_name_list = []  # 체크박스들의 상태를 저장할 리스트
    for i in CLASSES:
        if '-' in i:
            name = i.split('-')[0] + '_' + i.split('-')[1]
            checkbox_name_list.append(name)
        else:
            checkbox_name_list.append(i)
    num_checkboxes = len(CLASSES)  # 생성할 체크박스 개수
    checkbox_states = {}
    select_all = st.sidebar.checkbox("Select All")
    if select_all == False:
        for ind, i in enumerate(checkbox_name_list):
            num = ind
            checkbox_state = st.sidebar.checkbox(f"{i}", key=num, disabled=False)
            checkbox_states[num] = checkbox_state
    else:
        for ind, i in enumerate(checkbox_name_list):
            num = ind
            checkbox_state = st.sidebar.checkbox(f"{i}", key=num, disabled=True)
            checkbox_states[num] = checkbox_state

    col1, col2 = st.columns(2)
    with col1:
        time.sleep(1)
        if upload_file is not None:

            img = Image.open(upload_file)
            new_size = (2048, 2048)
            img = img.resize(new_size)
            img = img.save("img.jpg")
            img = cv2.imread("img.jpg")
            st.image(img)

            model = load_model()
            tf = A.Resize(512, 512)
            test_dataset = dataset.XRayInferenceDataset(transforms=tf, stream=True)
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=1,
                shuffle=False,
                num_workers=2,
                drop_last=False
            )
            rles, filename_and_class = test(model, test_loader)

    with col2:
        if upload_file is not None:
                preds = []
                for rle in rles[:len(CLASSES)]:
                    pred = decode_rle_to_mask(rle, height=2048, width=2048)
                    preds.append(pred)
                preds = np.stack(preds, 0)
                preds_size = preds.shape[1:] + (3, )
                array = np.zeros(preds_size, dtype=np.uint8)
                if select_all == False:
                    for checkbox_key, state in checkbox_states.items():
                        if state:
                            array += label2rgb(preds, checkbox_key)
                else:
                    array = all_label2rgb(preds)
                pred_image = Image.fromarray(array)
                st.image(pred_image)
        
if __name__ == '__main__':
    main()