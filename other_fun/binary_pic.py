import cv2
import numpy as np
from src_ele.dir_operation import traverseFolder
from src_ele.file_operation import get_ele_data_from_path
from src_ele.pic_opeeration import show_Pic
from src_ele.pic_spilt_ostu import OtsuFastMultithreshold

path_show = r'C:\Users\Administrator\Desktop\BBBB'

path_list = traverseFolder(path_show)
# print(path_list)


def alter_pic(img):
    img = 0.54 * img

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rand = 1+(np.random.random()-0.5)/2
            img[i][j] = int(img[i][j]*rand)
    return img


target_process_pic = path_list[2]

# 读取动态成像
print(target_process_pic)
img_dyna = 255-cv2.imread(target_process_pic, cv2.IMREAD_GRAYSCALE)
# img_dyna, _ = get_ele_data_from_path(target_process_pic)
# img_mask, _ = get_ele_data_from_path(target_process_pic.replace('dyna', 'mask'))
# img_stat, _ = get_ele_data_from_path(target_process_pic.replace('dyna', 'stat'))

img_dyna = cv2.resize(img_dyna, (250, 250))
# img_mask = cv2.resize(img_mask, (250, 250))
# img_stat = cv2.resize(img_stat, (250, 250))
# print('dyna stat mask shape is :{}, {}, {}'.format(img_dyna.shape, img_mask.shape, img_stat.shape))
# show_Pic([traverse_pic(img_dyna), (img_mask), traverse_pic(img_stat)], pic_order='13')


otsu = OtsuFastMultithreshold()
otsu.load_image(img_dyna)
kThresholds = otsu.calculate_k_thresholds(1)
kThresholds[0] = int(kThresholds[0]*1)
print('otsu thresholds is :{}'.format(kThresholds[0]))
# # exit(0)

crushed = otsu.apply_thresholds_to_image(kThresholds)
# crushed = img_dyna
# show_Pic([traverse_pic(img_dyna), (img_mask), traverse_pic(img_stat),crushed], pic_order='22')

# # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))            # 矩形
# # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))           # 交叉形
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 椭圆形
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3))  # 椭圆形
kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 椭圆形
kernel = kernel3

# cv2.MORPH_CLOSE 闭运算(close)，先膨胀后腐蚀的过程。闭运算可以用来排除小黑洞。
# cv2.MORPH_OPEN  开运算(open) ,先腐蚀后膨胀的过程。开运算可以用来消除小黑点，在纤细点处分离物体、平滑较大物体的边界的 同时并不明显改变其面积。
# iterations – 操作次数，默认为1
opening = cv2.morphologyEx(crushed, cv2.MORPH_OPEN, kernel, iterations=2)
closing = cv2.morphologyEx(crushed, cv2.MORPH_CLOSE, kernel, iterations=2)
# sure_bg = cv2.dilate(crushed, kernel, iterations=2)
show_Pic([255-img_dyna, crushed, opening, closing], pic_order='22')

# cv2.imwrite('test1.png', opening)


