import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import time
import cv2

import torch
from inference import encode_mask_to_rle
import inference

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

def main():
    st.set_page_config(layout="wide")
    st.title(":hand: Hand :bone: Bone :blue[Segmentation]")
    col1, col2 = st.columns(2)
    with col1:
        # menu = ['Image']
        upload_file = st.file_uploader('이미지를 업로드 하세요.', type=['png', 'jpg', 'jpeg'])
        time.sleep(1)
        if upload_file is not None:
            img = Image.open(upload_file)
            img = img.save("img.jpg")
            img = cv2.imread("img.jpg")
            st.image(img)
            # st.text(f'{image.shape}')
            image = np.array(img)
            st.text(f'{image.shape}')
            image = np.float32(image) / 255.
            st.text(f'{image.shape}')
            image.shape
            image = image.transpose(2, 0, 1) 
            image = np.expand_dims(image, axis=0)
            input_tensor = torch.from_numpy(image).float()
            st.text(f'input_tensor_{input_tensor.shape}')
            model = torch.load('/opt/ml/input/code/results_baseline/FPN_densenet169_150.pt')
            model.eval()
            # st.text(f'{model}')
            if torch.cuda.is_available():
                model = model.cuda()
                input_tensor = input_tensor.cuda()
            
            st.text(f'input_tensor_{input_tensor.shape}')
            output = model(input_tensor)
            st.text(f'output_{output}')
            output = torch.sigmoid(output)
            st.text(f'output__{output}')
            outputs = (outputs > 0.5).detach().cpu().numpy()
            rles = []
            filename_and_class = []
            for c, segm in enumerate(output):
                rle = encode_mask_to_rle(segm)
                rles.append(rle)
                filename_and_class.append(f"{IND2CLASS[c]}")
            st.text(f'{str(filename_and_class)}')
            


if __name__ == '__main__':
    main()