import cv2
from src_ele.pic_opeeration import show_Pic


path = r'D:\Data\GAN_EFFECT\simu_2\dyna_simu_1.png'

pic = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
print(pic.shape)


show_Pic([252-pic], pic_order='11')