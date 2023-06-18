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
from visualizatioin import label2rgb, decode_rle_to_mask
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

# @st.cache_data
# def infer(outputs):
#     return outputs

def test(model, data_loader, thr=0.5):
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = len(CLASSES)

        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()
            # images = Image.open(images)
            # images = images.unsqueeze(0)

            # outputs = model(images)['out']
            outputs = model(images)
            # outputs = run_model(_model=model, _inputs=images)
            
            # restore original size
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()
            # cache_outputs = infer(outputs)
            
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")

            # rles = infer(rles)
                    
    return rles, filename_and_class


def main():
    st.set_page_config(layout="wide")
    st.title(":hand: Hand :bone: Bone :blue[Segmentation]")
    st.sidebar.title('this is sidebar')
    # st.sidebar.checkbox('체크박스에 표시될 문구')
    upload_file = st.sidebar.file_uploader('이미지를 업로드 하세요.', type=['png', 'jpg', 'jpeg'])
    checkbox = st.sidebar.button('Segmentation!!')
    checkbox2 = st.sidebar.checkbox('Seg2')
    st.checkbox('Seg3')
    refresh_state = st.checkbox("Click me to toggle", key="refresh_checkbox")
    col1, col2 = st.columns(2)
    with col1:
        # menu = ['Image']
        # upload_file = st.file_uploader('이미지를 업로드 하세요.', type=['png', 'jpg', 'jpeg'])
        time.sleep(1)
        if upload_file is not None:

            img = Image.open(upload_file)
            img = img.save("img.jpg")
            img = cv2.imread("img.jpg")
            st.image(img)

            # model = torch.load('/opt/ml/input/weights/FPN_densenet161_150/FCN_resnet50_bce_loss_200_dataclean3_noacc_127e.pt')
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
            # rles = infer(rles)
    
            # preds = []
            # for rle in rles[:len(CLASSES)]:
            #     pred = decode_rle_to_mask(rle, height=2048, width=2048)
            #     preds.append(pred)
            # preds = np.stack(preds, 0)
            # fig, ax = plt.subplots(1, 2, figsize=(24, 12))
            # ax[0].imshow(img)    # remove channel dimension
            # ax[1].imshow(label2rgb(preds))
    with col2:
        if upload_file is not None and checkbox:
            preds = []
            for rle in rles[:len(CLASSES)]:
                pred = decode_rle_to_mask(rle, height=2048, width=2048)
                preds.append(pred)
            preds = np.stack(preds, 0)
            array = label2rgb(preds)
            pred_image = Image.fromarray(array)
            st.image(pred_image)
            # fig, ax = plt.subplots(1, 2, figsize=(24, 12))
            # ax[0].imshow(img)    # remove channel dimension
            # ax[1].imshow(label2rgb(preds))
            # st.pyplot(fig)
            # plt.show()
            # st.text(f'{image.shape}')
            # image = np.array(img)
            # st.text(f'{image.shape}')
            # image = np.float32(image) / 255.
            # st.text(f'{image.shape}')
            # image.shape
            # image = image.transpose(2, 0, 1) 
            # image = np.expand_dims(image, axis=0)
            # input_tensor = torch.from_numpy(image).float()
            # # st.text(f'input_tensor_{input_tensor.shape}')
            # model = torch.load('/opt/ml/input/weights/FPN_densenet161_150/FCN_resnet50_bce_loss_200_dataclean3_noacc_127e.pt')
            # model.eval()
    # st.sidebar.checkbox('체크박스에 표시될 문구')

if __name__ == '__main__':
    main()