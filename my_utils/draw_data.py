# -*- coding: utf-8 -*
# @Time : 2022/3/25 22:03
# @Author : 杨坤林
# @File : draw_data.py
# @Software : PyCharm

import seaborn as sns
import os
import matplotlib.pyplot as plt
import numpy as np



path_test = ""
pathall = "G:\\meta_data"

def draw(data_path, save_path):
    '''
    单独的画图函数，将npy文件保存到指定路径
    '''
    data = np.load(data_path)
    data = data.squeeze()
    plt.figure()
    sns.heatmap(data=data,
                cmap="gray_r"  # 淡色色盘：sns.light_palette()使用
                )
    plt.title(os.path.basename(data_path))
    # plt.show()
    plt.savefig(os.path.join(save_path, "{}.png".format(os.path.basename(data_path))))
    plt.close()


if __name__ == "__main__":
    # draw(path_test, 1)
    task_list = os.listdir(pathall)  # 加载出全部的任务列表
    for task in task_list:
        hr_temp_path = os.path.join(pathall, task, "hr")
        lr_temp_path = os.path.join(pathall, task, "lr")
        hr_list = os.listdir(hr_temp_path)  # 低分、高分文件列表
        lr_list = os.listdir(lr_temp_path)
        hr_temp_save_path = os.path.join(pathall, task, "hr_image")  # 低分、高分文件保存路径
        lr_temp_save_path = os.path.join(pathall, task, "lr_image")
        if not os.path.isdir(hr_temp_save_path):  # 判断是否存在，不存在，创建
            os.makedirs(hr_temp_save_path)
        if not os.path.isdir(lr_temp_save_path):  # 判断是否存在，不存在，创建
            os.makedirs(lr_temp_save_path)
        # 遍历文件，并画图保存
        for file in hr_list:
            temp_path1 = os.path.join(hr_temp_path, file)
            draw(temp_path1, hr_temp_save_path)
        for file in lr_list:
            temp_path2 = os.path.join(lr_temp_path, file)
            draw(temp_path2, lr_temp_save_path)
        print("over+1!")

