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

def app():
    st.title("Data EDA")
    st.markdown("---")
    tab1, tab2= st.tabs(['Graph1' , 'Graph2'])
    with tab1:
        img = cv2.imread("/opt/ml/input/level2_cv_semanticsegmentation-cv-11/apps/polygon.JPG")
        st.image(img)
    with tab2:
        img2 = cv2.imread("/opt/ml/input/level2_cv_semanticsegmentation-cv-11/apps/polygon2.JPG")
        st.image(img2)

