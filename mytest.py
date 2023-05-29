# -*- coding: utf-8 -*
# @Time : 2022/2/28 20:37
# @Author : 杨坤林
# @File : mytest.py
# @Software : PyCharm
from __future__ import print_function
import argparse
import os
import numpy as np
from PIL import Image
import torch
import math
from os.path import join
import torchvision.transforms as transforms
import random
from collections import OrderedDict
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import normalized_root_mse as compare_nrmse
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
import my_RDN_modle as my_modle
import mydataset
from torch.autograd import Variable
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import mean_absolute_error
import torchvision.utils as vutils


parser = argparse.ArgumentParser(description='my meta_sr test step')

parser.add_argument('--nFeat', type=int, default=64, help='number of feature maps')
parser.add_argument('--nChannel', type=int, default=3, help='number of color channels to use')
parser.add_argument('--scale', type=int, default=2, help='scale output size /input size')
parser.add_argument('--growthRate', type=int, default=32, help='growthRate of dense net')
parser.add_argument('--nDenselayer', type=int, default=6, help='nDenselayer of RDB')

parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
parser.add_argument('--modle_dir', type=str, default="./train_model_350")  # 预训练模型地址
parser.add_argument('--my_train_logs_path', type=str, default="./my_test")   # 输出日志保存地址
parser.add_argument('--dataset', type=str, default='G:/data/meta_test_data/', help='data path')
parser.add_argument('--my_save_path', type=str, default='./my_test', help='data path')

save_file_name = "test_summary_CH5_TEMP_IRWVP.csv"

# 创建字典
def build_summary_dict(total_losses, psnr, phase, summary_losses=None):

    if summary_losses is None:
        summary_losses = {}

    summary_losses["{}_loss_mean".format(phase)] = np.nanmean(total_losses)
    summary_losses["{}_psnr_mean".format(phase)] = np.nanmean(psnr)

    return summary_losses

# 创建保存路径
def create_exp_dir(exp):
    if not os.path.isdir(exp):  # 判断是否存在，不存在，创建
        os.makedirs(exp)

# 设置优化器
def set_lr(args, epoch, optimizer):
    lrDecay = args.lrDecay
    decayType = args.decayType
    if decayType == 'step':
        epoch_iter = (epoch + 1) // lrDecay
        lr = args.lr / 2**epoch_iter
    elif decayType == 'exp':
        k = math.log(2) / lrDecay
        lr = args.lr * math.exp(-k * epoch)
    elif decayType == 'inv':
        k = 1 / lrDecay
        lr = args.lr / (1 + k * epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



def my_normal(x):
    smax = np.max(x)
    smin = np.min(x)
    s = (x - smin)/(smax - smin)
    return s


# 测试单任务模型
def my_test2():
    args = parser.parse_args()
    create_exp_dir(args.my_train_logs_path)
    # 设置随机数种子
    manualSeed = 101
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    print("Random Seed: ", manualSeed)
    # # dataset 和 dataloader
    # my_test_dataset = mydataset.myDataset(args.train_data_dir)
    # my_test_dataloader = mydataset.get_dataloader(my_test_dataset, args.train_batch_size)
    # 加载预训练模型
    state_dict = torch.load(args.modle_dir)
    # state_dict_loaded = state['network']  # 还可以加载 state['optimizer']
    # print(state_dict_loaded)
    # 这里使用ordered dict
    # names_weights_copy = OrderedDict({
    #     name.replace('sr_modle.', ''): value for
    #     name, value in state_dict_loaded.items()})
    device = torch.device("cuda:0" if args.cuda else "cpu")
    modle = my_modle.RDN(args)
    # 这里的strict=False 非常方便，会自己硬塞，名字相同的就塞进去，没有的就不管

    modle.load_state_dict(state_dict['modle'])
    modle.eval()

    # 这里的代码是为了调试时，看是否权值赋值成功
    # weight_show = modle.named_parameters()
    # weights_toshow = {
    #     name: value for
    #     name, value in weight_show}

    # 创建test.csv记录loss
    df = pd.DataFrame(columns=['data_name', 'test Loss', 'test psnr'])  # 列名
    df.to_csv(os.path.join(args.my_train_logs_path, "test_summary.csv"), index=False)  # 路径可以根据需要更改


    # 放到GPU中
    modle.to(device)

    image_dir = args.dataset

    lr_dir = join(image_dir, "lr")
    hr_dir = join(image_dir, "hr")

    image_names = os.listdir(lr_dir)
    i = 0
    for image_name in image_names:
        lr = np.load(join(lr_dir, image_name))
        hr = np.load(join(hr_dir, image_name))

        # 归一化
        arr_no_0 = lr.flatten()[np.flatnonzero(lr)]
        lr_mean = arr_no_0.mean()
        lr_std = arr_no_0.std()
        lr = np.where(lr == 0, lr_mean, lr)
        lr = (lr - lr_mean) / lr_std
        lr = np.expand_dims(lr, 0)
        lr_normal = my_normal(lr)
        lr = torch.from_numpy(np.expand_dims(lr, 0))



        arr_no_01 = hr.flatten()[np.flatnonzero(hr)]
        hr_mean = arr_no_01.mean()
        hr_std = arr_no_01.std()
        hr = np.where(hr == 0, hr_mean, hr)
        hr = (hr - hr_mean) / hr_std
        hr = np.expand_dims(hr, 0)
        hr = torch.from_numpy(np.expand_dims(hr, 0))

        modle.eval()

        im_lr = Variable(lr.cuda(), volatile=False)
        im_hr = Variable(hr.cuda())
        with torch.no_grad():
            output = modle(im_lr)
        loss = F.smooth_l1_loss(output, im_hr)

        output = output.detach().cpu().numpy()
        im_hr = im_hr.detach().cpu().numpy()
        im_hr_normal = my_normal(im_hr)
        output_normal = my_normal(output)

        psnr = sk_psnr(im_hr_normal, output_normal)
        create_exp_dir(os.path.join(args.my_save_path, 'input'))
        create_exp_dir(os.path.join(args.my_save_path, 'truth'))
        create_exp_dir(os.path.join(args.my_save_path, 'output'))
        np.save(os.path.join(args.my_save_path, 'input', '%d.npy' % i), lr_normal)
        np.save(os.path.join(args.my_save_path, 'truth', '%d.npy' % i), im_hr_normal)
        np.save(os.path.join(args.my_save_path, 'output', '%d.npy' % i), output_normal)
        i+=1

        loss1 = "%f" % loss
        psnr1 = "%f" % psnr
        data_name = "%s" % image_name

        list = [data_name, loss1, psnr1]

        data = pd.DataFrame([list])
        data.to_csv(os.path.join(args.my_train_logs_path, "test_summary.csv"), mode='a', header=False,
                    index=False)  # mode设为a,就可以向csv文件追加数据了

        # 输出参数
        log = "data:{} \t loss: {:.4f}\t psnr: {:.4f}".format(
            image_name,
            loss,
            psnr)
        print(log)



# 测试MMTCSR效果
def my_test3():
    args = parser.parse_args()
    create_exp_dir(args.my_train_logs_path)
    # 设置随机数种子
    manualSeed = 101
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    print("Random Seed: ", manualSeed)
    # # dataset 和 dataloader
    # my_test_dataset = mydataset.myDataset(args.train_data_dir)
    # my_test_dataloader = mydataset.get_dataloader(my_test_dataset, args.train_batch_size)
    # 加载预训练模型
    state = torch.load(args.modle_dir)
    state_dict_loaded = state['network']  # 还可以加载 state['optimizer']
    # print(state_dict_loaded)
    # 这里使用ordered dict
    names_weights_copy = OrderedDict({
        name.replace('sr_modle.', ''): value for
        name, value in state_dict_loaded.items()})
    device = torch.device("cuda:0" if args.cuda else "cpu")
    modle = my_modle.RDN(args)
    # 这里的strict=False 非常方便，会自己硬塞，名字相同的就塞进去，没有的就不管
    modle.load_state_dict(names_weights_copy, strict=False)
    # modle.load_state_dict(state_dict['modle'])
    modle.eval()



    # 创建test.csv记录loss
    df = pd.DataFrame(columns=['data_name', 'psnr', 'rmse', 'SSIM', 'mae'])  # 列名
    df.to_csv(os.path.join(args.my_train_logs_path, save_file_name), index=False)  # 路径可以根据需要更改


    # 放到GPU中
    modle.to(device)

    image_dir = args.dataset

    lr_dir = join(image_dir, "lr")
    hr_dir = join(image_dir, "hr")
    matched_dir = join(image_dir, "matched_lr")
    matched_dir2 = join(image_dir, 'matched_lr2')

    image_names = os.listdir(lr_dir)
    i = 0

    psnr_all = []
    ssim_all = []
    rmse_all = []
    mae_all = []
    for image_name in image_names:
        lr = np.load(join(lr_dir, image_name))
        hr = np.load(join(hr_dir, image_name))
        matched = np.load(join(matched_dir, image_name))
        matched2 = np.load(join(matched_dir2, image_name))
        # lr归一化

        lr_normal = (lr - 100) / (350 - 100)

        # hr 归一化

        hr_normal = (hr - 100) / (350 - 100)

        # matched 归一化

        matched_normal = (matched - 100) / (350 - 100)
        matched2_normal = (matched2 - 100) / (350 - 100)


        a = torch.from_numpy(np.expand_dims(lr_normal, 0))
        # hr = torch.from_numpy(np.expand_dims(hr_normal, 0))
        c = torch.from_numpy(np.expand_dims(matched_normal, 0))
        e = torch.from_numpy(np.expand_dims(matched2_normal, 0))
        d = torch.concat([a, c, e], dim=0)
        input_lr = d.float()

        input_lr = input_lr.unsqueeze(0)
        # hr = hr.unsqueeze(0)

        modle.eval()

        im_lr = Variable(input_lr.cuda(), volatile=False)
        # im_hr = Variable(hr.cuda())
        with torch.no_grad():
            output = modle(im_lr)
        # loss = F.l1_loss(output, im_hr)

        output = output.detach().cpu().numpy().squeeze()
        # im_hr = im_hr.detach().cpu().numpy()
        # im_hr_normal = my_normal(im_hr)
        # output_normal = my_normal(output)

        # 归一化
        output = np.where(output > 1, 1, output)
        output = np.where(output < 0, 0, output)

        hr_normal = np.where(hr_normal > 1, 1, hr_normal)
        hr_normal = np.where(hr_normal < 0, 0, hr_normal)
        # hrmask = hr_normal!=1
        # output = output*hrmask
        # output = np.where(output==0,1,output)
        # output = my_normal(output)
        # hr_normal = my_normal(hr_normal)

        # 反归一化
        out_img_inv = output * (350 - 100) + 100


        psnr = sk_psnr(hr_normal, output)
        ssim = sk_ssim(hr_normal.squeeze(), output.squeeze())
        mse = compare_mse(hr_normal, output)
        rmse = mse**0.5
        mae = mean_absolute_error(hr_normal, output)

        psnr_all.append(psnr)
        ssim_all.append(ssim)
        rmse_all.append(rmse)
        mae_all.append(mae)

        create_exp_dir(os.path.join(args.my_save_path, 'input'))
        create_exp_dir(os.path.join(args.my_save_path, 'truth'))
        create_exp_dir(os.path.join(args.my_save_path, 'output'))
        np.save(os.path.join(args.my_save_path, 'input', '%s.npy' % image_name), lr)
        np.save(os.path.join(args.my_save_path, 'truth', '%s.npy' % image_name), hr)
        np.save(os.path.join(args.my_save_path, 'output', '%s.npy' % image_name), out_img_inv)
        i+=1

        mae1 = "%f" % mae
        psnr1 = "%f" % psnr
        ssim1 = "%f" % ssim
        rmse1 = "%f" % rmse
        data_name = "%s" % image_name

        list = [data_name, psnr1, rmse1, ssim1, mae1]

        data = pd.DataFrame([list])
        data.to_csv(os.path.join(args.my_train_logs_path, save_file_name), mode='a', header=False,
                    index=False)  # mode设为a,就可以向csv文件追加数据了

        # 输出参数
        # log = "data:{} \t loss: {:.4f}\t psnr: {:.4f}".format(
        #     image_name,
        #     loss,
        #     psnr)
        # print(log)
    mae2 = "%f" % np.mean(mae_all)
    psnr2 = "%f" % np.mean(psnr_all)
    ssim2 = "%f" % np.mean(ssim_all)
    rmse2 = "%f" % np.mean(rmse_all)
    data_name2 = "%s" % 'average'
    list2 = [data_name2, psnr2, rmse2, ssim2, mae2]
    data2 = pd.DataFrame([list2])
    data2.to_csv(os.path.join(args.my_train_logs_path, save_file_name), mode='a', header=False,
                index=False)
    print(psnr2)




if __name__ == '__main__':
    print("start!")
    my_test3()
    print("over!")
