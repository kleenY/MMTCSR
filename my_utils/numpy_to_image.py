# -*- coding: utf-8 -*
# @Time : 2022/5/6 22:46
# @Author : 杨坤林
# @File : numpy_to_image.py
# @Software : PyCharm

import os
import random
import time

import cv2 as cv
from PIL import Image
from tqdm import tqdm
import seaborn as sns
import os
import matplotlib.pyplot as plt
import numpy as np



def draw(data_path):

    data = np.load(data_path)
    data = data.squeeze()
    plt.figure()
    sns.heatmap(data=data,
                cmap="gray_r"
                )
    plt.title(os.path.basename(data_path))

    plt.savefig("new_test.png")
    plt.close()

def convert1(path_test):

    image_array = np.load(path_test)

    arr_no_0 = image_array.flatten()[np.flatnonzero(image_array)]
    max = arr_no_0.max()
    min = arr_no_0.min()

    image_array = np.where(image_array == 0, max, image_array)
    image_array = (image_array - min) / (max - min)


    image_array = 1 - image_array
    image_array *= 255  # 变换为0-255的灰度值

    im = Image.fromarray(image_array)
    im1 = im.convert('F')
    print(im1.getpixel((255, 255)))
    print(type(im1.getpixel((255, 255))))
    im2 = im.convert('L')
    print(im2.getpixel((255, 255)))
    print(type(im2.getpixel((255, 255))))
    im3 = im.convert('I')
    print(im3.getpixel((255, 255)))
    print(type(im3.getpixel((255, 255))))

    im1.save('test1.tiff')
    im2.save('test2.png')
    im3.save('test3.png')

def convert2():

    randomByteArray = bytearray(os.urandom(120000))

    flatNumpyArray = np.array(randomByteArray)

    grayImage = flatNumpyArray.reshape(300, 400)

    cv.imshow('GrayImage', grayImage)

    print(grayImage)
    cv.waitKey()


    randomByteArray1 = bytearray(os.urandom(360000))
    flatNumpyArray1 = np.array(randomByteArray1)
    BGRimage = flatNumpyArray1.reshape(300, 400, 3)
    cv.imshow('BGRimage', BGRimage)
    cv.waitKey()
    cv.destroyAllWindows()


def pbar():
    pbar = tqdm(total=100)
    for i in range(100):
        time.sleep(1)
        pbar.update(1)


def convert3(path, savepath):

    image_array = np.load(path)

    image_array = (image_array - 100) / (350 - 100)

    image_array = 1 - image_array


    image_array *= 255

    im = Image.fromarray(image_array)

    im2 = im.convert('L')
    name = os.path.basename(path)
    name = name[:-8]
    name = name + '.png'
    temp_savepath = os.path.join(savepath, name)


    im2.save(temp_savepath)



def convert_all(npy_path, img_path):
    data_list = os.listdir(npy_path)

    for data in data_list:
        temp_path = os.path.join(npy_path, data)
        convert3(temp_path, img_path)



if __name__ == '__main__':
    print("start!")

    path_all = ''
    model_list = os.listdir(path_all)
    pbar = tqdm(total=len(model_list))
    for model_name in model_list:
        temp_model_path = os.path.join(path_all, model_name)
        temp_npy_path = os.path.join(temp_model_path, 'npy')
        temp_img_path = os.path.join(temp_model_path, 'image')
        task_list = os.listdir(temp_npy_path)
        for task_name in task_list:
            npy_path = os.path.join(temp_npy_path, task_name)
            save_path = os.path.join(temp_img_path, task_name)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            convert_all(npy_path, save_path)

        pbar.update(1)
    pbar.close()

    print("over!")


