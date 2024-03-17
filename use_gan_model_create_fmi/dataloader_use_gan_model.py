import random
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from pic_preprocess import pic_binary_random
from src_ele.dir_operation import traverseFolder
from src_ele.file_operation import get_ele_data_from_path
from src_ele.pic_opeeration import show_Pic




class ImageDataset_FMI_SIMULATE_LAYER(Dataset):
    def __init__(self, path=r'D:\GitHubProj\dino\fracture_simulation\background_simulate\fracture_moni_8.png', x_l=256, y_l=256, step=10, win_len=400):
        super().__init__()
        self.mask_full = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        print(self.mask_full.shape)

        self.length = (self.mask_full.shape[0]-win_len)//step + 1

        self.x_l = x_l
        self.y_l = y_l
        self.step = step
        self.win_len = win_len

    def __getitem__(self, index):
        mask = self.mask_full[index*self.step:index*self.step+self.win_len, :]
        # print(mask.shape)

        mask = cv2.resize(mask, (self.x_l, self.y_l))

        mask = mask.reshape((1, self.x_l, self.y_l))/256
        mask = np.append(mask, mask, axis=0)

        return mask
        # return pic_all_org, pic_all_mask


    def __len__(self):
        return self.length


# a = ImageDataset_FMI_SIMULATE_LAYER()
# for i in range(10):
#     b = a[i]
#     print(b.shape)
#     show_Pic([b[0,:,:]], pic_order='11')
