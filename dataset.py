import os
import h5py
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

class CelebA_hq_DataLoader(Dataset):
    def __init__(self, size, h5py_path, data_dir_path, txtpath, transform=None):
        with open(txtpath, 'r') as f:
            txtlist = f.readlines()
            
        self.data_dir_path = data_dir_path
        self.img_name_list = txtlist
        self.transform = transform
        
    def __len__(self):
        return len(self.img_name_list)
    
    def __getitem__(self, i):
        img = Image.open(
            os.path.join(
                self.data_dir_path, 
                self.img_name_list[i].split()[0]))
        
        if self.transform:
            img = self.transform(img)
        return img
    
class FFHQ_DataLoader(Dataset):
    def __init__(self, size, data_dir_path, transform=None):
        ffhq_item_list = sorted(os.listdir(data_dir_path))
        Ims_pathes = []
        for dir_name in ffhq_item_list:
            img_names = os.listdir(os.path.join(data_dir_path, dir_name))
            for img_name in img_names:
                Ims_pathes.append(os.path.join(data_dir_path, dir_name, img_name))
                
        self.Ims_pathes = Ims_pathes
        self.transform = transform
        print(len(self.Ims_pathes))
        
    def __len__(self):
        return len(self.Ims_pathes)
    
    def __getitem__(self, i):
        img = Image.open(self.Ims_pathes[i])
        
        if self.transform:
            img = self.transform(img)
        return img
        
