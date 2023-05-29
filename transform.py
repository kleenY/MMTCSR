# -*- coding: utf-8 -*
# @Time : 2022/10/3 17:07
# @Author : 杨坤林
# @File : transform.py
# @Software : PyCharm

import numpy as np
import random

# 垂直翻转
def flip_v(source):
    test_h = np.copy(source)

    return test_h[::-1]

# 旋转180度
def flip180(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    return new_arr

# 水平翻转
def flip_h(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    test_v = new_arr[::-1]

    return test_v

# 旋转270度（逆时针旋转90度）

def flip90_left(arr):
    new_arr = np.transpose(arr)
    new_arr = new_arr[::-1]
    return new_arr

# 旋转90度
def flip90_right(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    new_arr = np.transpose(new_arr)[::-1]
    return new_arr


class Numpy_Transform(object):
    def __init__(self, mode='train', probability=0.5):
        self.mode = mode
        self.probability = probability

    def __call__(self, sample):
        if self.mode == 'train':
            if round(np.random.uniform(0, 1), 1) <= self.probability:
                image = sample
                img_aug = random.choice([flip_v, flip180, flip_h, flip90_right, flip90_left])(image)
                return img_aug
            else:
                return sample

        if self.mode == 'test' or self.mode == 'infer':
            return sample


if __name__ == '__main__':

    # 原始数组

    arr0 = np.load('G:\\meta_data\\train\\CH3_TEMP_IRSPL\\lr\\0.npy')
    # arr0 = np.expand_dims(arr0, 0)
    print(arr0.size)
    print('原始数组：\n', arr0)
    flip_180 = flip180(arr0)
    left_90 = flip90_left(arr0)
    right_90 = flip90_right(arr0)
    flip_v = flip_v(arr0)
    flip_h = flip_h(arr0)
    print('===== flip_180 ====\n',flip_180,'\n')
    print('===== left_90 =====\n',left_90,'\n')
    print('===== right_90 =====\n',right_90,'\n')
    print('===== flip_v =====\n',flip_v,'\n')
    print('===== flip_h =====\n', flip_h, '\n')
