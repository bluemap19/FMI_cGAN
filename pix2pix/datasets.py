import glob
import random
import os

import cv2
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

from pic_preprocess import pic_binary_random
from src_ele.dir_operation import traverseFolder
from src_ele.file_operation import get_ele_data_from_path
from src_ele.pic_opeeration import show_Pic


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        if mode == "train":
            self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.files)


class ImageDataset_FMI(Dataset):
    def __init__(self, path=r'D:\Data\target_uns_class', x_l=256, y_l=256, pic_rorate=True):
        super().__init__()
        self.list_all_file = traverseFolder(path)
        self.length = len(self.list_all_file)//2
        self.x_l = x_l
        self.y_l = y_l
        self.pic_rorate = pic_rorate

    def __getitem__(self, index):
        path_temp = self.list_all_file[index*2]
        path_temp_stat = ''
        # print(path_temp)

        if path_temp.__contains__('dyna'):
            path_temp_stat = path_temp.replace('dyna', 'stat')
        elif path_temp.__contains__('stat'):
            path_temp_stat = path_temp
            path_temp = path_temp_stat.replace('stat', 'dyna')

        pic_dyna, depth = get_ele_data_from_path(path_temp)
        pic_stat, depth = get_ele_data_from_path(path_temp_stat)
        pic_dyna = cv2.resize(pic_dyna, (self.x_l, self.y_l))
        pic_stat = cv2.resize(pic_stat, (self.x_l, self.y_l))

        # show_Pic([pic_dyna, pic_stat], pic_order='12', pic_str=['', ''], save_pic=False, path_save='')

        # dyna_kThreshold_shift = np.random.randint(22, 28)
        dyna_kThreshold_shift = 1.3
        pic_dyna_mask, pic_dyna = pic_binary_random(pic_dyna, kThreshold_shift=dyna_kThreshold_shift, pic_rorate=self.pic_rorate)
        # stat_kThreshold_shift = np.random.randint(22, 28)
        stat_kThreshold_shift = 1.2
        pic_stat_mask, pic_stat = pic_binary_random(pic_stat, kThreshold_shift=stat_kThreshold_shift, pic_rorate=self.pic_rorate)
        # print('kThreshold_shift dyna and stat is:{}, {}'.format(dyna_kThreshold_shift, stat_kThreshold_shift))

        pic_dyna = pic_dyna.reshape((1, self.x_l, self.y_l))/256
        pic_stat = pic_stat.reshape((1, self.x_l, self.y_l))/256
        pic_all_org = np.append(pic_dyna, pic_stat, axis=0)

        pic_dyna_mask = pic_dyna_mask.reshape((1, self.x_l, self.y_l))/256
        pic_stat_mask = pic_stat_mask.reshape((1, self.x_l, self.y_l))/256
        pic_all_mask = np.append(pic_dyna_mask, pic_stat_mask, axis=0)

        return {"A": pic_all_org, "B": pic_all_mask}
        # return pic_all_org, pic_all_mask


    def __len__(self):
        return self.length


# a = ImageDataset_FMI(r'D:\Data\ele_img_big_small_mix_GAN_test', pic_rorate=True)
# for i in range(10):
#     b = a[np.random.randint(0, a.length-1)]
#     print(b['A'].shape, b['B'].shape)
#     show_Pic([1-b['A'][0,:,:], 1-b['A'][1,:,:], b['B'][0,:,:], b['B'][1,:,:]], pic_order='22')
