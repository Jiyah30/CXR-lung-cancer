import torchvision.transforms as T
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import pandas as pd
from PIL import Image
import os
import torch
import multiprocessing
import random
from sklearn.model_selection import train_test_split
import torchxrayvision as xrv
import skimage


# Third-party packages

# Normalize with ImageNet mean and std
# STD = np.array([0.229, 0.224, 0.225])
STD = np.array([1])
# MEAN = np.array([0.485, 0.456, 0.406])
MEAN = np.array([0])

class RandomChoice(torch.nn.Module):
    def __init__(self, transforms):
       super().__init__()
       self.transforms = transforms
       self.always_apply = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
            # xrv.datasets.XRayCenterCrop(),
            # xrv.datasets.XRayResizer(224)
        ])

    def __call__(self, imgs):
        t = random.choice(self.transforms)
        if t is not None:
            return [self.always_apply(t(img)) for img in imgs]
        else:
            return [self.always_apply(img) for img in imgs]

def train_transforms():
    return [
        T.RandomRotation(10),
        T.RandomHorizontalFlip(),
        # T.GaussianBlur(7,3),
        # T.RandomAffine(degrees=(-30,30), translate=(0, 0.5), scale=(0.4, 0.5), shear=(0,0), fillcolor=(0,255,255)),
        # T.ColorJitter(brightness=(0, 5), contrast=(
        # 0, 5), saturation=(0, 5), hue=(-0.1, 0.1)),
        None,
    ]

def normalize(img):
    img = img - img.min()
    img = img / img.max()
    return img

def get_transform(train):
    return RandomChoice([None]) if train else RandomChoice([None])

class CXRDataset(torch.utils.data.Dataset):

    def __init__(self, root, annotations_df, transforms):
        
        self.root = root
        self.df = annotations_df
        self.transforms = transforms
        
    def __getitem__(self, x):

        row = self.df.iloc[x, :].values

        filepath_before, filepath_after, partial_response, progressive_disease, stable_disease = row

        # before_image = skimage.io.imread(os.path.join(self.root, filepath_before), as_gray=True)
        # after_image = skimage.io.imread(os.path.join(self.root, filepath_after), as_gray=True)
        # before_image = xrv.datasets.normalize(before_image, 255)
        # after_image = xrv.datasets.normalize(after_image, 255)
        # if self.transforms is not None:
        #     before_image, after_image = self.transforms([before_image[None], after_image[None]])

        before_image = np.array(Image.open(os.path.join(self.root, filepath_before)))
        after_image = np.array(Image.open(os.path.join(self.root, filepath_after)))

        before_image = normalize(before_image)
        after_image = normalize(after_image)
        before_image = torch.from_numpy(before_image)
        after_image = torch.from_numpy(after_image)
        before_image = before_image.permute(2, 0, 1)
        after_image = after_image.permute(2, 0, 1)
        # before_image = torch.from_numpy(before_image)[None]
        # after_image = torch.from_numpy(after_image)[None]

        label = (partial_response==1)*0 + (progressive_disease==1)*1 + (stable_disease==1)*2
        
        return before_image, after_image, torch.tensor(label)
    
    def __len__(self):
        return self.df.shape[0]
    
def get_dataloader(base_root, csv_name, batch_size, seed=42):

    train_transform = get_transform(train=True)
    val_transform = get_transform(train=False)

    annotations_df = pd.read_csv(os.path.join(base_root, csv_name))

    train_indices, val_indices = train_test_split(range(annotations_df.shape[0]), test_size=0.2, random_state=seed)
    val_indices, test_indices = val_indices[:int(len(val_indices)/2)], val_indices[int(len(val_indices)/2):]

    tarin_dataset = CXRDataset(base_root, annotations_df.iloc[train_indices], train_transform)
    val_dataset = CXRDataset(base_root, annotations_df.iloc[val_indices], val_transform)
    test_dataset = CXRDataset(base_root, annotations_df.iloc[test_indices], val_transform)

    # num_workers = multiprocessing.Pool()._processes

    trainloader =  torch.utils.data.DataLoader(tarin_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    validloader =  torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    testloader =  torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    return trainloader, validloader, testloader