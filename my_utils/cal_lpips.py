# -*- coding: utf-8 -*
# @Time : 2022/11/24 20:29
# @Author : 杨坤林
# @File : cal_lpips.py
# @Software : PyCharm
from tqdm import tqdm
import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
# import cv2
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='PSNR SSIM script', add_help=False)
parser.add_argument('--input_images_path', default='/media/aita-ocean/data/YKL/meta_test_data/image_hr')
parser.add_argument('--image2smiles2image_save_path', default='/home/aita-ocean/Documents/YKL/meta_tcsr_3/meta_tcsr_3.1_6/my_test/image0')
parser.add_argument('-v', '--version', type=str, default='0.1')
parser.add_argument('--my_train_logs_path', default='./lpips')
parser.add_argument('--save_file_name', default='lpips1.csv')
args = parser.parse_args()


def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg", ".png", ".jpeg"])


# def load_img(filepath):
#     img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
#     img = img.astype(np.float32)
#     img = img / 255.
#     return img



loss_fn = lpips.LPIPS(net='alex', version=args.version)

# 创建保存路径
def create_exp_dir(exp):
    if not os.path.isdir(exp):  # 判断是否存在，不存在，创建
        os.makedirs(exp)


if __name__ == '__main__':


    create_exp_dir(args.my_train_logs_path)
    df = pd.DataFrame(columns=['data_name', 'lpips'])  # 列名
    df.to_csv(os.path.join(args.my_train_logs_path, args.save_file_name), index=False)  # 路径可以根据需要更改

    files = os.listdir(args.input_images_path)
    i = 0
    total_lpips_distance = 0
    average_lpips_distance = 0
    # lpips_list = []
    for file in files:

        try:
            # Load images
            img0 = lpips.im2tensor(lpips.load_image(os.path.join(args.input_images_path, file)))
            img1 = lpips.im2tensor(lpips.load_image(os.path.join(args.image2smiles2image_save_path, file)))

            if (os.path.exists(os.path.join(args.input_images_path, file)),
                os.path.exists(os.path.join(args.image2smiles2image_save_path, file))):
                i = i + 1

            # Compute distance
            current_lpips_distance = loss_fn.forward(img0, img1)
            # lpips_list.append(current_lpips_distance)

            data_name = "%s" % file
            lpips1 = "%f" % current_lpips_distance
            list = [data_name,lpips1]
            data = pd.DataFrame([list])
            data.to_csv(os.path.join(args.my_train_logs_path, args.save_file_name), mode='a', header=False,
                        index=False)  # mode设为a,就可以向csv文件追加数据了

            total_lpips_distance = total_lpips_distance + current_lpips_distance

        except Exception as e:
            print(e)

    average_lpips_distance = float(total_lpips_distance) / i
    data_name2 = "%s" % 'average'
    lpips2 = "%f" % average_lpips_distance
    list2 = [data_name2, lpips2]
    data2 = pd.DataFrame([list2])
    data2.to_csv(os.path.join(args.my_train_logs_path, args.save_file_name), mode='a', header=False,
                 index=False)

    print("The processed iamges is ", i)
    print("LPIPS: %f " % (average_lpips_distance))
