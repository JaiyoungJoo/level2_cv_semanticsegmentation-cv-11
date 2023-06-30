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
from visualization import label2rgb, decode_rle_to_mask, all_label2rgb
import albumentations as A
from torch.utils.data import Dataset, DataLoader
import dataset
from model import Encoder
import base64

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
    model = torch.load('/opt/ml/input/weights/FPN_densenet161_150/FPN_densenet169_150.pt')
    model2 = torch.load('/opt/ml/input/weights/serving/for_serving_multimodal_multitask_120_1024_noseedup.pt')
    model3 = Encoder(2048).cuda()
    state_dict = torch.load('/opt/ml/input/weights/serving/deep_svdd.pth')
    model3.load_state_dict(state_dict['net_dict'])
    c = torch.Tensor(state_dict['center']).cuda()
    model.eval()
    model2.eval()
    model3.eval()
    return model, model2, model3, c

def eval(net, c, dataloader):
   # ROC AUC score 계산
    net.eval()
    with torch.no_grad():
        for x in dataloader:
            x = x.float().cuda()
            z = net.encoder(x)
            z = torch.flatten(z, 0)
            z = net.latent(z)
            score = torch.sum((z - c) ** 2, dim=0)
    return score

# @st.cache(allow_output_mutation=True)
# def infer(outputs):
#     return outputs

meta_info = {'age_min' :19 , 'age_denominator' : 69 - 19,
    'weight_min' : 42, 'weight_denominator' : 118 - 42,
    'height_min' : 150, 'height_denominator' : 187 - 150}

# def test(model, data_loader, thr=0.5):
#     model = model.cuda()
#     model.eval()

#     rles = []
#     filename_and_class = []
#     with torch.no_grad():
#         n_class = len(CLASSES)

#         for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
#             images = images.cuda()

#             # outputs = model(images)['out']
#             outputs = model(images)
 
#             # restore original size
#             outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
#             outputs = torch.sigmoid(outputs)
#             outputs = (outputs > thr).detach().cpu().numpy()
   
#             for output, image_name in zip(outputs, image_names):
#                 for c, segm in enumerate(output):
#                     rle = encode_mask_to_rle(segm)
#                     rles.append(rle)
#                     filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
   
#     return rles, filename_and_class

def test(model, meta_model, anomaly_model, c, data_loader,data_loader2,data_loader3, thr=0.5,tta_enabled=False):
    model = model.cuda()
    meta_model.cuda()
    anomaly_model.cuda()
    model.eval()
    meta_model.eval()
    anomaly_model.eval()
    c.cuda()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = len(CLASSES)

        for step, ((images, image_names),(images2,_),(images3,_)) in tqdm(enumerate(zip(data_loader,data_loader2,data_loader3)), total=len(data_loader)):
            images = images.cuda()
            images2 = images2.cuda()
            images3 = images3.cuda()
            outputs = model(images)
            meta_out = meta_model.net.backbone(images2)
            ap = F.adaptive_avg_pool2d(meta_out,(1,1))
            out_age = meta_model.net.branch_age(ap)
            out_gender = meta_model.net.branch_gender(ap)
            out_weight = meta_model.net.branch_weight(ap)
            out_height = meta_model.net.branch_height(ap)
            
            out_age = int(float(out_age) * meta_info['age_denominator'] + meta_info['age_min'])
            if out_gender[0,0] >= out_gender[0,1]:
                out_gender = 'male'
            else:
                out_gender = 'female'
            out_weight = int(float(out_weight) * meta_info['weight_denominator'] + meta_info['weight_min'])
            out_height = int(float(out_height) * meta_info['height_denominator'] + meta_info['height_min'])

            
            # restore original size
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()
            
            for output, image_name in zip(outputs, image_names):
                for c1, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c1]}_{image_name}")
            
        score = eval(anomaly_model, c, images3)
        if score>= 24.7:  
            check_anomal= True
        else:
            check_anomal =False
                    
    return rles, filename_and_class, out_age, out_gender, out_weight, out_height, score, check_anomal

def addline(image):
    image_array = np.array(image)

    # Add red border to the image array
    border_width = 15
    border_color = (255, 0, 0)  # Red color
    image_array[0:border_width, :] = border_color
    image_array[-border_width:, :] = border_color
    image_array[:, 0:border_width] = border_color
    image_array[:, -border_width:] = border_color

    # Convert the modified image array back to PIL image
    modified_image = Image.fromarray(image_array)

    return modified_image

def app():
    st.title(":hand: Hand :bone: Bone :blue[Segmentation]")
    st.markdown("---")
    upload_file = st.file_uploader('이미지를 업로드 하세요.', type=['png', 'jpg', 'jpeg'])

    st.sidebar.markdown("---")
    st.sidebar.title('Select part of bone')
    all=st.sidebar.checkbox("Select all")
    states={}
    if all:
        select_multi_bones = st.sidebar.multiselect(
    '확인하고자 하는 뼈 부분을 선택해 주세요.\n\n복수선택가능',CLASSES,CLASSES)
    else:
        select_multi_bones = st.sidebar.multiselect(
    '확인하고자 하는 뼈 부분을 선택해 주세요.\n\n복수선택가능',CLASSES)
        
    seg = st.sidebar.button('Done')
        # st.sidebar.markdown(
        #     """
        #     <style>
        #     .css-1pd56a0{
        #             border: 1px solid #f8f8f8;
        #             background-color: #f8f8f8;
        #             border-radius: 5px;
        #         }
        #     </style>
        #     """,
        #     unsafe_allow_html=True,
        # )
        
    st.sidebar.markdown("---")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            time.sleep(1)
            if upload_file is not None:
                img = Image.open(upload_file)
                new_size = (2048, 2048)
                img = img.resize(new_size)
                img = img.save("img.jpg")
                img = cv2.imread("img.jpg")
                text = "Original Image"
                styled_text = f"<h3 style='text-align: center;'>{text}</h3>"
                st.markdown(styled_text, unsafe_allow_html=True)
                st.image(img)

                # model = load_model()
                # tf = A.Resize(512, 512)
                # test_dataset = dataset.XRayInferenceDataset(transforms=tf, stream=True)
                # test_loader = DataLoader(
                #     dataset=test_dataset, 
                #     batch_size=1,
                #     shuffle=False,
                #     num_workers=2,
                #     drop_last=False
                # )
                # rles, filename_and_class = test(model, test_loader)

        with col2:
            if upload_file is not None and seg:
                with st.spinner('Loading..'):
                    model, model2, model3, c = load_model()
                    tf1 = A.Resize(512, 512)
                    tf2 = A.Resize(1024, 1024)
                    tf3 = A.Resize(2048, 2048)
                    test_dataset1 = dataset.XRayInferenceDataset(transforms=tf1, stream=True)
                    test_dataset2 = dataset.XRayInferenceDataset(transforms=tf2, stream=True)
                    test_dataset3 = dataset.XRayInferenceDataset(transforms=tf3, stream=True)
                    test_loader1 = DataLoader(
                        dataset=test_dataset1, 
                        batch_size=1,
                        shuffle=False,
                        num_workers=2,
                        drop_last=False
                    )
                    test_loader2 = DataLoader(
                        dataset=test_dataset2, 
                        batch_size=1,
                        shuffle=False,
                        num_workers=2,
                        drop_last=False
                    )
                    test_loader3 = DataLoader(
                        dataset=test_dataset3, 
                        batch_size=1,
                        shuffle=False,
                        num_workers=2,
                        drop_last=False
                    )
                    
                    rles, filename_and_class, out_age, out_gender, out_weight, out_height, score, check_anomal = test(model, model2,model3,c,test_loader1,test_loader2,test_loader3)
                    preds = []
                    for rle in rles[:len(CLASSES)]:
                        pred = decode_rle_to_mask(rle, height=2048, width=2048)
                        preds.append(pred)
                    preds = np.stack(preds, 0)
                    preds_size = preds.shape[1:] + (3, )
                    array = np.zeros(preds_size, dtype=np.uint8)
                    if all:
                        array = all_label2rgb(preds)
                    else:
                        for i in range(len(select_multi_bones)):
                            if select_multi_bones[i]:
                                array+=label2rgb(preds,i)
                    pred_image = Image.fromarray(array)
                    text = "Bone Image"
                    styled_text = f"<h3 style='text-align: center;'>{text}</h3>"
                    st.markdown(styled_text, unsafe_allow_html=True)
                    if check_anomal==True:
                        pred_image=addline(pred_image)
                        st.image(pred_image)
                        
                    else:
                        st.image(pred_image)
                            
                if check_anomal==False:
                    if out_gender=='male':
                        st.text(f'1️⃣ 나이: {out_age} 2️⃣ 성별: 👨  3️⃣ 몸무게: {out_weight}, 4️⃣ 키: {out_height}')
                    else:
                        st.text(f'1️⃣ 나이: {out_age} 2️⃣ 성별: 👩  3️⃣ 몸무게: {out_weight}, 4️⃣ 키: {out_height}')
                    st.subheader("정상입니다 😀")
                else:
                    if score>10000:
                        st.subheader("이미지를 다시 업로드 하세요")
                    else:
                        if out_gender=='male':
                            st.text(f'1️⃣ 나이: {out_age} 2️⃣ 성별: 👨  3️⃣ 몸무게: {out_weight}, 4️⃣ 키: {out_height}')
                        else:
                            st.text(f'1️⃣ 나이: {out_age} 2️⃣ 성별: 👩  3️⃣ 몸무게: {out_weight}, 4️⃣ 키: {out_height}')
                        st.subheader("병원을 방문해보세요 🤕")
                    
            
                
        st.markdown(
            """
            <style>
            div[data-testid="stHorizontalBlock"]{
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
