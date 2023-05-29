# -*- codeing = utf-8 -*-
# @Time : 2021/12/13 22:14
# @Author : 杨坤林
# @File : dataextract.py
# @Software : PyCharm



import os
import numpy as np
import xarray as xr
import cv2

pathall = "G:\\test\\avhrr"

savepath = "G:\\test1\\CH5_TEMP_IRSPL"

def listdir(path):
    namelist = []
    for file in os.listdir(path):
        namelist.append(file)
    return namelist


def convnpy(path,filnameb1,filnameavh,savepath):
    fileb1 = os.path.join(path, filnameb1)
    fileavh = os.path.join(path, filnameavh)
    dataset2 = xr.open_dataset(fileavh)
    # print("*" * 20)
    ch5 = dataset2.variables['CH5_TEMP'].values
    # print(ch5)
    # print("*" * 20)
    # print(fileb1)
    dataset1 = xr.open_dataset(fileb1)
    irwin = dataset1.variables['IRSPL'].values  # 注意他是(1, 301, 301)
    if not os.path.isdir(savepath):  # 判断是否存在，不存在，创建
        os.makedirs(savepath)

    ch5 = np.array(ch5)
    #print(ch5)
    #print("*" * 20)
    irwin = np.array(irwin)

    irwin = np.squeeze(irwin)

    sch5 = cv2.resize(ch5, dsize=(301, 301), interpolation=cv2.INTER_CUBIC)

    maskavh = sch5 > 0

    newirwin = maskavh * irwin

    ch5[np.isnan(ch5)] = 0
    # print(ch5)
    # print(ch5.shape)
    nameb1 = "maskedirwin_" + filnameb1
    namemask = "maskofch5_" + filnameavh
    path1 = os.path.join(savepath,nameb1)
    path2 = os.path.join(savepath,filnameavh)
    path3 = os.path.join(savepath,filnameb1)
    path4 = os.path.join(savepath,namemask)
    np.save(path1, newirwin)
    np.save(path2, ch5)
    np.save(path3, irwin)
    np.save(path4, maskavh)
    print("save!")

def mainwork(pathall,savepath):
    filelist = listdir(pathall)
    filelist.sort()
    for file in filelist:
        sfilepath = os.path.join(pathall,file)
        sfilelist = listdir(sfilepath)
        a = sfilelist[0].split(".")[9]


        if (a == 'hursat-b1'):
            sfileb1 = sfilelist[0]
            sfileavh = sfilelist[1]
        else:
            sfileb1 = sfilelist[1]
            sfileavh = sfilelist[0]

        esavepath = os.path.join(savepath, file)
        try:
            convnpy(sfilepath, sfileb1, sfileavh, esavepath)
        except:
            continue







if __name__=="__main__":
    print("start!")
    mainwork(pathall, savepath)
    print("end!")
