# -*- coding: utf-8 -*
# @Time : 2022/3/20 20:15
# @Author : 杨坤林
# @File : heatmap2.py
# @Software : PyCharm
# -*- codeing = utf-8 -*-
# @Time : 2021/12/31 11:32
# @Author : 杨坤林
# @File : heatmap.py
# @Software : PyCharm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import seaborn as sns
import os
from tqdm import tqdm
#import palettable #python颜色库



path = "./testlr.npy"
pathoutput = "../test_result/pred/0_0_sr.npy"
pathtruth = "../test_result/truth/0_0_truth.npy"
pathinput = "G:\\test2\\b1\\8230.npy"
save_path = "../my_test/image"

path_pred = "../test_result/pred"
path_truth = "../test_result/truth"
path_input = "../test_result/input"

def main():
    # 加载数据
    data_output = np.load(pathoutput).squeeze()
    data_input = np.load(pathinput).squeeze()
    data_truth = np.load(pathtruth).squeeze()
    data_error = abs(data_output - data_truth)

    print(data_error.shape)



    f = plt.figure(1)
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)


    sns.heatmap(data=data_input,
                ax = ax1,
                cmap = "gray_r"
               )
    ax1.set_title("input")

    sns.heatmap(data=data_truth,
                ax=ax2,
                cmap="gray_r"
                )
    ax2.set_title("truth")

    sns.heatmap(data=data_output,
                ax=ax3,
                cmap="gray_r"
                )
    ax3.set_title("output")

    sns.heatmap(data=data_error,
                ax=ax4,
                cmap="gray_r"
                )
    ax4.set_title("error")

    f.suptitle(os.path.basename(pathinput))
    plt.show()




def main2():
    # 加载数据
    data_output = np.load(pathoutput).squeeze()

    data_truth = np.load(pathtruth).squeeze()
    data_error = abs(data_output - data_truth)

    print(data_error.shape)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25*4, 25))


    sns.heatmap(data=data_truth,
                ax=ax2,
                cmap="gray_r"
                )
    ax2.set_title("truth")

    sns.heatmap(data=data_output,
                ax=ax1,
                cmap="gray_r"  # 淡色色盘：sns.light_palette()使用
                )
    ax3.set_title("output")

    sns.heatmap(data=data_error,
                ax=ax3,
                cmap="gray_r"  # 淡色色盘：sns.light_palette()使用
                )
    ax3.set_title("error")

    # f.suptitle(os.path.basename(pathinput))
    # plt.show()
    plt.savefig(os.path.join(save_path, "{}.png".format(os.path.basename(pathoutput))))
    plt.close()


def main3(path_pred, path_truth, path_input):
    # 加载数据
    data_output = np.load(path_pred).squeeze()
    # data_input = np.load(pathinput).squeeze()
    data_truth = np.load(path_truth).squeeze()
    data_input = np.load(path_input).squeeze()
    data_error = abs(data_output - data_truth)

    # print(data_error.shape)

    fig, axs = plt.subplots(2, 2, figsize=(7, 6))

    sns.heatmap(data=data_input,
                ax=axs[0, 0],
                cmap="gray_r"  # 淡色色盘：sns.light_palette()使用
                )
    axs[0, 0].set_title("input")
    axs[0, 0].axis('off')

    sns.heatmap(data=data_output,
                ax=axs[0, 1],
                cmap="gray_r"  # 淡色色盘：sns.light_palette()使用
                )
    axs[0, 1].set_title("output")
    axs[0, 1].axis('off')

    sns.heatmap(data=data_truth,
                ax=axs[1, 0],
                cmap="gray_r"  # 淡色色盘：sns.light_palette()使用
                )
    axs[1, 0].set_title("truth")
    axs[1, 0].axis('off')

    sns.heatmap(data=data_error,
                ax=axs[1, 1],
                cmap="gray_r"  # 淡色色盘：sns.light_palette()使用
                )
    axs[1, 1].set_title("error")

    fig.suptitle(os.path.basename(path_truth))
    # plt.show()
    plt.axis('off')
    plt.savefig(os.path.join(save_path, "{}.png".format(os.path.basename(path_pred))))
    plt.close()

def draw_all():
    list_pred = os.listdir(path_pred)
    # list_truth = os.listdir(path_truth)
    for id_pred in list_pred:
        pred_path = os.path.join(path_pred, id_pred)
        input_path = os.path.join(path_input, id_pred.split("_")[0]+"_"+id_pred.split("_")[1]+"_"+id_pred.split("_")[2]+("_lr.npy"))
        truth_path = os.path.join(path_truth, id_pred.split("_")[0]+"_"+id_pred.split("_")[1]+"_"+id_pred.split("_")[2]+("_truth.npy"))
        main3(pred_path, truth_path, input_path)
        print("over!")

def draw_all2():
    name_list = os.listdir("../my_test/output")
    pbar = tqdm(total=len(name_list))
    if not os.path.isdir('../my_test/image'):  # 判断是否存在，不存在，创建
        os.makedirs('../my_test/image')
    for name in name_list:
        output_path = os.path.join('../my_test/output', name)
        input_path = os.path.join('../my_test/input', name)
        truth_path = os.path.join('../my_test/truth', name)
        main3(output_path, truth_path, input_path)
        pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    draw_all2()