
from torch.utils.data import Dataset
import numpy as np
import os
import cv2
import torch


IMAGE_ROOT = "/home/supergalaxy/junha/input/data/train/DCM/"
LABEL_ROOT = "/home/supergalaxy/junha/input/data/train/outputs_json"
TEST_ROOT = "/home/supergalaxy/junha/input/data/test/DCM"

# abnormal data
ABNORMAL_PNGS = [
    'ID073/image1661736368856.png', 'ID073/image1661736410568.png',
    'ID124/image1661910068358.png', 'ID124/image1661910096458.png',
    'ID288/image1664156956152.png', 'ID288/image1664156981600.png',
    'ID363/image1664935962797.png', 'ID375/image1665452681174.png',
    'ID387/image1665536734529.png', 'ID387/image1665536751182.png',
    'ID430/image1666059865789.png', 'ID430/image1666059889440.png',
    'ID487/image1666661955150.png', 'ID506/image1666747096906.png',
    'ID519/image1666749288019.png', 'ID519/image1666749315607.png',
    'ID523/image1667178735444.png', 'ID523/image1667178762956.png',
    'ID543/image1667266674012.png', 'ID543/image1667266700981.png',
]

class AEDataset(Dataset):
    def __init__(self, transforms=None):
        pngs = {
        os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT) # relpath : 상대 경로로 변경
        for root, _dirs, files in os.walk(IMAGE_ROOT)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
        }
        print("#########################################################")
        print("Data Cleaning....")
        print("#########################################################")
        pngs = sorted(list(set(pngs) - set(ABNORMAL_PNGS)))
        _filenames = np.array(pngs)
        
        self.filenames = _filenames
        self.transforms = transforms
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):

        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)
        image = cv2.imread(image_path)
        image = image / 255.
        
        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # make channel first
        
        image = torch.from_numpy(image).float()
            
        return image
    
    
class AE_Test_Dataset(Dataset):
    def __init__(self, transforms=None):
        pngs = sorted(list(ABNORMAL_PNGS))
        _filenames = np.array(pngs)
        
        self.filenames = _filenames
        self.transforms = transforms
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):

        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)
        image = cv2.imread(image_path)
        image = image / 255.
        
        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # make channel first
        
        image = torch.from_numpy(image).float()
            
        return image
