
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import GroupKFold
import os
import cv2
import json
import torch
import pandas as pd
import albumentations as A
import albumentations.pytorch

# 데이터 경로를 입력하세요
IMAGE_ROOT = "/opt/ml/input/data/train/DCM/"
LABEL_ROOT = "/opt/ml/input/data/train/outputs_json/"
TEST_ROOT = "/opt/ml/input/data/test/DCM/"

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
ABNORMAL_JSONS = [
    'ID073/image1661736368856.json', 'ID073/image1661736410568.json',
    'ID124/image1661910068358.json', 'ID124/image1661910096458.json',
    'ID288/image1664156956152.json', 'ID288/image1664156981600.json',
    'ID363/image1664935962797.json', 'ID375/image1665452681174.json',
    'ID387/image1665536734529.json', 'ID387/image1665536751182.json',
    'ID430/image1666059865789.json', 'ID430/image1666059889440.json',
    'ID487/image1666661955150.json', 'ID506/image1666747096906.json',
    'ID519/image1666749288019.json', 'ID519/image1666749315607.json',
    'ID523/image1667178735444.json', 'ID523/image1667178762956.json',
    'ID543/image1667266674012.json', 'ID543/image1667266700981.json',
]

def check_size_of_dataset(IMAGE_ROOT, LABEL_ROOT):
    pngs = {
        os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT) # relpath : 상대 경로로 변경
        for root, _dirs, files in os.walk(IMAGE_ROOT)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
        }
    if LABEL_ROOT:
        jsons = {
            os.path.relpath(os.path.join(root, fname), start=LABEL_ROOT)
            for root, _dirs, files in os.walk(LABEL_ROOT)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".json"
            }
        jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
    pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}

    if LABEL_ROOT:
        assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
        assert len(pngs_fn_prefix - jsons_fn_prefix) == 0
        jsons = sorted(jsons)

    pngs = sorted(pngs)

    if LABEL_ROOT:
        return pngs, jsons
    else:
        return pngs

def equalize_and_remove_black(image, **kwargs):
    # 평활화 적용
    image = cv2.convertScaleAbs(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)

    # 검은색 부분 제거
    mask = equalized > 200
    result = np.zeros_like(image)
    result[mask] = image[mask]

    return result

def clahe(image, **kwargs):
    image = cv2.convertScaleAbs(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.uint8)
    # CLAHE 적용
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(image)
    rgb_image = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2RGB)
    return rgb_image
    
def clahe2(image, **kwargs):
    # Albumentations의 CLAHE 적용
    image = image.astype(np.uint8)
    transform = A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), always_apply=True)
    transformed_image = transform(image=image)["image"]
    return transformed_image

def get_transform():
    train_transform = [
        A.Resize(1024,1024),
        A.ElasticTransform(p=0.5, alpha=300, sigma=20, alpha_affine=50),
        A.Rotate(limit=45),
        A.RandomContrast(limit=[0,0.5],p=1)
    ]
    val_transform = [
        A.Resize(1024,1024),
    ]

    return A.Compose(train_transform), A.Compose(val_transform)

class XRayInferenceDataset(Dataset):
    def __init__(self, transforms=None, stream=False):
        pngs = check_size_of_dataset(TEST_ROOT, False)
        if stream:
            pngs = 'img.jpg'
        _filenames = pngs
        _filenames = np.array(sorted(_filenames))
        
        self.filenames = _filenames
        self.transforms = transforms
        self.stream = stream
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        # image_name = self.filenames[item]
        # image_path = os.path.join(TEST_ROOT, image_name)
        if self.stream:
            image_name = 'img.jpg'
            image_path = '/opt/ml/level2_cv_semanticsegmentation-cv-11/' + image_name
        else: 
            image_name = self.filenames[item]
            image_path = os.path.join(TEST_ROOT, image_name)
        image = cv2.imread(image_path)
        image = image / 255.
        
        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # make channel first
        
        image = torch.from_numpy(image).float()
            
        return image, image_name

class XRayDataset(Dataset):
    def __init__(self, is_train=True, transforms=None, seed = 21, dataclean = None):
        pngs, jsons = check_size_of_dataset(IMAGE_ROOT, LABEL_ROOT)
        self.dataclean = dataclean
        if self.dataclean:
            print("#########################################################")
            print("Data Cleaning....")
            print("#########################################################")
            pngs = sorted(list(set(pngs) - set(ABNORMAL_PNGS)))
            jsons = sorted(list(set(jsons) - set(ABNORMAL_JSONS)))
        _filenames = np.array(pngs)
        _labelnames = np.array(jsons)
        
        # split train-valid
        # 한 폴더 안에 한 인물의 양손에 대한 `.dcm` 파일이 존재하기 때문에
        # 폴더 이름을 그룹으로 해서 GroupKFold를 수행합니다.
        # 동일 인물의 손이 train, valid에 따로 들어가는 것을 방지합니다.
        groups = [os.path.dirname(fname) for fname in _filenames]
        
        # dummy label
        ys = [0 for fname in _filenames]
        
        # 전체 데이터의 20%를 validation data로 쓰기 위해 `n_splits`를
        # 5으로 설정하여 KFold를 수행합니다.
        gkf = GroupKFold(n_splits=5)
        
        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if is_train:
                # 0번을 validation dataset으로 사용합니다.
                if i == (seed+4)%5:
                    continue
                    
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])
            
            else:
                filenames = list(_filenames[y])
                labelnames = list(_labelnames[y])
                
                # skip i > 0
                break

        # self.dataclean = dataclean
        self.filenames = filenames
        # if self.dataclean:
        #     self.filenames = list(set(self.filenames) - set(ABNORMAL_PNGS))
        self.labelnames = labelnames
        # if self.dataclean:
        #     self.labelnames = list(set(self.labelnames) - set(ABNORMAL_JSONS))
        #     print("#########################################################")
        #     print("Data Cleaning....")
        #     print('filenames', len(self.filenames))
        #     print('labelnames', len(self.labelnames))
        #     print("#########################################################")
        self.is_train = is_train
        self.transforms = transforms
   
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)

        image = cv2.imread(image_path)
        image = image / 255.
        
        label_name = self.labelnames[item]
        label_path = os.path.join(LABEL_ROOT, label_name)
        
        # process a label of shape (H, W, NC)
        label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)
        
        # read label file
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]
        
        # iterate each class
        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])
            
            # polygon to mask
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
        
        if self.transforms is not None:
            # if any(isinstance(t, A.CLAHE) for t in self.transforms.transforms):
            #     image = image.astype(np.uint8)  # 데이터 타입을 uint8로 변환
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)
            
            # if any(isinstance(t, A.CLAHE) for t in self.transforms.transforms):
            # image = result["image"].astype(np.float32)
            # else:
            image = result["image"]

            label = result["mask"] if self.is_train else label

        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # make channel first
        label = label.transpose(2, 0, 1)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
            
        return image, label
    

class XRayDataset_path(Dataset):
    def __init__(self, is_train=True, transforms=None, seed = 21):
        pngs, jsons = check_size_of_dataset(IMAGE_ROOT, LABEL_ROOT)

        _filenames = np.array(pngs)
        _labelnames = np.array(jsons)
        
        # split train-valid
        # 한 폴더 안에 한 인물의 양손에 대한 `.dcm` 파일이 존재하기 때문에
        # 폴더 이름을 그룹으로 해서 GroupKFold를 수행합니다.
        # 동일 인물의 손이 train, valid에 따로 들어가는 것을 방지합니다.
        groups = [os.path.dirname(fname) for fname in _filenames]
        
        # dummy label
        ys = [0 for fname in _filenames]
        
        # 전체 데이터의 20%를 validation data로 쓰기 위해 `n_splits`를
        # 5으로 설정하여 KFold를 수행합니다.
        gkf = GroupKFold(n_splits=5)
        
        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if is_train:
                # 0번을 validation dataset으로 사용합니다.
                if i == (seed+4)%5:
                    continue
                    
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])
            
            else:
                filenames = list(_filenames[y])
                labelnames = list(_labelnames[y])
                
                # skip i > 0
                break
        
        self.filenames = filenames
        self.labelnames = labelnames
        self.is_train = is_train
        self.transforms = transforms
       
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)
        
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        label_name = self.labelnames[item]
        label_path = os.path.join(LABEL_ROOT, label_name)
        
        # process a label of shape (H, W, NC)
        label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)
        
        # read label file
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]
        
        # iterate each class
        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])
            
            # polygon to mask
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
        
        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)
            
            image = result["image"]
            label = result["mask"] if self.is_train else label

        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # make channel first
        label = label.transpose(2, 0, 1)
        
        image = torch.from_numpy(image).float().half()
        label = torch.from_numpy(label).float().half()
            
        return image, label, image_path    
    
class XRayDataset_valid(Dataset):
    def __init__(self, is_train=True, transforms=None, seed = 21):
        pngs, jsons = check_size_of_dataset(IMAGE_ROOT, LABEL_ROOT)

        _filenames = np.array(pngs)
        _labelnames = np.array(jsons)
        
        # split train-valid
        # 한 폴더 안에 한 인물의 양손에 대한 `.dcm` 파일이 존재하기 때문에
        # 폴더 이름을 그룹으로 해서 GroupKFold를 수행합니다.
        # 동일 인물의 손이 train, valid에 따로 들어가는 것을 방지합니다.
        groups = [os.path.dirname(fname) for fname in _filenames]
        
        # dummy label
        ys = [0 for fname in _filenames]
        
        # 전체 데이터의 20%를 validation data로 쓰기 위해 `n_splits`를
        # 5으로 설정하여 KFold를 수행합니다.
        gkf = GroupKFold(n_splits=5)
        
        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if is_train:
                # 0번을 validation dataset으로 사용합니다.
                if i == (seed+4)%5:
                    continue
                    
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])
            
            else:
                filenames = list(_filenames[y])
                labelnames = list(_labelnames[y])
                
                # skip i > 0
                break
        
        self.filenames = filenames
        self.labelnames = labelnames
        self.is_train = is_train
        self.transforms = transforms
       
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)
        
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        label_name = self.labelnames[item]
        label_path = os.path.join(LABEL_ROOT, label_name)
        
        # process a label of shape (H, W, NC)
        label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)
        
        # read label file
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]
        
        # iterate each class
        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])
            
            # polygon to mask
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
        
        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)
            
            image = result["image"]
            label = result["mask"] if self.is_train else label

        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # make channel first
        label = label.transpose(2, 0, 1)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
            
        return image, image_name, label   

class XRayDataset_gray(Dataset):
    def __init__(self, is_train=True, transforms=None):
        pngs, jsons = check_size_of_dataset(IMAGE_ROOT, LABEL_ROOT)
        
        _filenames = np.array(pngs)
        _labelnames = np.array(jsons)

        # split train-valid
        # 한 폴더 안에 한 인물의 양손에 대한 `.dcm` 파일이 존재하기 때문에
        # 폴더 이름을 그룹으로 해서 GroupKFold를 수행합니다.
        # 동일 인물의 손이 train, valid에 따로 들어가는 것을 방지합니다.
        groups = [os.path.dirname(fname) for fname in _filenames]
        
        # dummy label
        ys = [0 for fname in _filenames]
        
        # 전체 데이터의 20%를 validation data로 쓰기 위해 `n_splits`를
        # 5으로 설정하여 KFold를 수행합니다.
        gkf = GroupKFold(n_splits=5)
        
        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if is_train:
                # 0번을 validation dataset으로 사용합니다.
                if i == 0:
                    continue
                    
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])
            
            else:
                filenames = list(_filenames[y])
                labelnames = list(_labelnames[y])
                
                # skip i > 0
                break
        
        self.filenames = filenames
        self.labelnames = labelnames
        self.is_train = is_train
        self.transforms = transforms
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)
        
        image = cv2.imread(image_path, flags=cv2.IMREAD_GRAYSCALE)
        image = np.expand_dims(image, axis=-1)
        image = image / 255.
        
        label_name = self.labelnames[item]
        label_path = os.path.join(LABEL_ROOT, label_name)
        
        # process a label of shape (H, W, NC)
        label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)
        
        # read label file
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]
        
        # iterate each class
        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])
            
            # polygon to mask
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
        
        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)
            
            image = result["image"]
            label = result["mask"] if self.is_train else label

        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # make channel first
        label = label.transpose(2, 0, 1)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
        label = torch.logical_not(torch.eq(torch.sum(label, dim=0, keepdim=True), 0)).float()
            
        return image, label

class XRayInferenceDataset_gray(Dataset):
    def __init__(self, transforms=None):
        pngs = check_size_of_dataset(TEST_ROOT, False)
        
        _filenames = pngs
        _filenames = np.array(sorted(_filenames))
        
        self.filenames = _filenames
        self.transforms = transforms
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(TEST_ROOT, image_name)
        
        image = cv2.imread(image_path, flags=cv2.IMREAD_GRAYSCALE)
        image = np.expand_dims(image, axis=-1)
        image = image / 255.
        
        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # make channel first
        
        image = torch.from_numpy(image).float()
            
        return image, image_name







class XRayDataset_Multi(Dataset):
    def __init__(self, is_train=True, transforms=None, seed = 21):
        pngs, jsons = check_size_of_dataset(IMAGE_ROOT, LABEL_ROOT)

        _filenames = np.array(pngs)
        _labelnames = np.array(jsons)
        
        # split train-valid
        # 한 폴더 안에 한 인물의 양손에 대한 `.dcm` 파일이 존재하기 때문에
        # 폴더 이름을 그룹으로 해서 GroupKFold를 수행합니다.
        # 동일 인물의 손이 train, valid에 따로 들어가는 것을 방지합니다.
        groups = [os.path.dirname(fname) for fname in _filenames]
        
        # dummy label
        ys = [0 for fname in _filenames]
        
        # 전체 데이터의 20%를 validation data로 쓰기 위해 `n_splits`를
        # 5으로 설정하여 KFold를 수행합니다.
        gkf = GroupKFold(n_splits=5)
        
        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if is_train:
                # 0번을 validation dataset으로 사용합니다.
                if i == (seed+4)%5:
                    continue
                    
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])
            
            else:
                filenames = list(_filenames[y])
                labelnames = list(_labelnames[y])
                
                # skip i > 0
                break
        
        self.filenames = filenames
        self.labelnames = labelnames
        self.is_train = is_train
        self.transforms = transforms
        self.meta = pd.read_excel('/opt/ml/input/data/meta_data.xlsx')
        self.age_max = 69
        self.age_min = 19
        self.age_denominator = self.age_max - self.age_min
        self.weight_max = 118
        self.weight_min = 42
        self.weight_denominator = self.weight_max - self.weight_min
        self.hight_max = 187
        self.hight_min = 150
        self.hight_denominator = self.hight_max - self.hight_min
       
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)
        id_num = int(image_path.split('/')[-2][-3:])
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        label_name = self.labelnames[item]
        label_path = os.path.join(LABEL_ROOT, label_name)
        
        # process a label of shape (H, W, NC)
        label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)
        
        # read label file
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]
        
        # iterate each class
        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])
            
            # polygon to mask
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
        
        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)
            
            image = result["image"]
            label = result["mask"] if self.is_train else label

        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # make channel first
        label = label.transpose(2, 0, 1)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
        
        info = self.meta[self.meta['ID'] == id_num]
        
        age = torch.tensor([(int(info['나이'].iloc[0])-self.age_min)/self.age_denominator]).float()
        gender = torch.tensor([1,0]).float() if str(info['성별'].iloc[0]).split('_')[-1] == '남' else torch.tensor([0,1]).float()
        weight = torch.tensor([(int(info['체중(몸무게)'].iloc[0])-self.weight_min)/self.weight_denominator]).float()
        hight = torch.tensor([(int(info['키(신장)'].iloc[0])-self.hight_min)/self.hight_denominator]).float()
        return image, label, age, gender, weight, hight
    


class XRayInferenceDataset_Multi(Dataset):
    def __init__(self, transforms=None):
        pngs = check_size_of_dataset(TEST_ROOT, False)
        
        _filenames = pngs
        _filenames = np.array(sorted(_filenames))
        
        self.filenames = _filenames
        self.transforms = transforms
        self.meta = pd.read_excel('/opt/ml/input/data/meta_data.xlsx')
        self.age_max = 69
        self.age_min = 19
        self.age_denominator = self.age_max - self.age_min
        self.weight_max = 118
        self.weight_min = 42
        self.weight_denominator = self.weight_max - self.weight_min
        self.hight_max = 187
        self.hight_min = 150
        self.hight_denominator = self.hight_max - self.hight_min
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(TEST_ROOT, image_name)
        id_num = int(image_path.split('/')[-2][-3:])
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # make channel first
        
        image = torch.from_numpy(image).float()

        info = self.meta[self.meta['ID'] == id_num]
        
        age = torch.tensor([(int(info['나이'].iloc[0])-self.age_min)/self.age_denominator]).float()
        gender = torch.tensor([1,0]).float() if str(info['성별'].iloc[0]).split('_')[-1] == '남' else torch.tensor([0,1]).float()
        weight = torch.tensor([(int(info['체중(몸무게)'].iloc[0])-self.weight_min)/self.weight_denominator]).float()
        hight = torch.tensor([(int(info['키(신장)'].iloc[0])-self.hight_min)/self.hight_denominator]).float()
            
        return image, image_name, age, gender, weight, hight