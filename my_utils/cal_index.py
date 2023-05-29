# -*- coding: utf-8 -*
# @Time : 2022/11/21 16:53
# @Author : 杨坤林
# @File : cal_index.py
# @Software : PyCharm

from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import normalized_root_mse as compare_nrmse
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
import os
from PIL import Image
save_file_name = "test_summary6.csv"
my_train_logs_path = "./index"
path_1 = "/media/aita-ocean/data/YKL/meta_test_data/image_lr"
path_2 = "/media/aita-ocean/data/YKL/meta_test_data/target"
"/home/aita-ocean/Documents/YKL/meta_tcsr_3/meta_tcsr_3.0_6/my_test/image0_gradient"

def create_exp_dir(exp):
    if not os.path.isdir(exp):
        os.makedirs(exp)

def cal_index():
    create_exp_dir(my_train_logs_path)

    df = pd.DataFrame(columns=['data_name', 'psnr', 'rmse', 'SSIM', 'mae', 'correlation'])
    df.to_csv(os.path.join(my_train_logs_path, save_file_name), index=False)

    data_list = os.listdir(path_1)
    psnr_all = []
    rmse_all = []
    mae_all = []
    ssim_all = []
    cor_all = []
    for data_name in data_list:
        temp_data1_path = os.path.join(path_1, data_name)
        temp_data2_path = os.path.join(path_2, data_name)
        data1 = np.array(Image.open(temp_data1_path))/255.0
        data2 = np.array(Image.open(temp_data2_path))/255.0

        psnr = sk_psnr(data1, data2)
        ssim = sk_ssim(data1.squeeze(), data2.squeeze())
        mse = compare_mse(data1, data2)
        rmse = mse**0.5
        mae = mean_absolute_error(data1, data2)
        cor = np.corrcoef(data1.flatten(), data2.flatten())
        cor = cor[0, 1]

        psnr_all.append(psnr)
        ssim_all.append(ssim)
        rmse_all.append(rmse)
        mae_all.append(mae)
        cor_all.append(cor)



        mae1 = "%f" % mae
        psnr1 = "%f" % psnr
        ssim1 = "%f" % ssim
        rmse1 = "%f" % rmse
        data_name = "%s" % data_name
        cor1 = "%f" % cor

        list = [data_name, psnr1, rmse1, ssim1, mae1, cor1]

        data = pd.DataFrame([list])
        data.to_csv(os.path.join(my_train_logs_path, save_file_name), mode='a', header=False,
                    index=False)


    mae2 = "%f" % np.mean(mae_all)
    psnr2 = "%f" % np.mean(psnr_all)
    ssim2 = "%f" % np.mean(ssim_all)
    rmse2 = "%f" % np.mean(rmse_all)
    cor2 = "%f" % np.mean(cor_all)
    data_name2 = "%s" % 'average'
    list2 = [data_name2, psnr2, rmse2, ssim2, mae2, cor2]
    data2 = pd.DataFrame([list2])
    data2.to_csv(os.path.join(my_train_logs_path, save_file_name), mode='a', header=False,
                index=False)
    print(psnr2)

if __name__ == '__main__':
    print("start!")
    cal_index()
    print("over!")