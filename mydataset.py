# -*- codeing = utf-8 -*-
# @Time : 2022/2/18 11:22
# @Author : 杨坤林
# @File : mydataset.py
# @Software : PyCharm
from os import listdir
from os.path import join
import random

from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms



import numpy as np
from PIL import Image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    img = img.resize((256, 256), Image.BICUBIC)
    return img

class myDataset(data.Dataset):
    def __init__(self, image_dir):
        super(myDataset, self).__init__()
        # self.direction = direction
        self.a_path = join(image_dir, "lr")
        self.b_path = join(image_dir, "hr")
        self.image_filenames = [x for x in listdir(self.a_path)]  # 因为a和b的名字相同



    def __getitem__(self, index):
        try:
            a = np.load(join(self.a_path, self.image_filenames[index]))
            b = np.load(join(self.b_path, self.image_filenames[index]))

            arr_no_0 = a.flatten()[np.flatnonzero(a)]
            lr_mean = arr_no_0.mean()
            lr_std = arr_no_0.std()
            a = np.where(a == 0, lr_mean, a)
            a = (a - lr_mean) / lr_std

            arr_no_01 = b.flatten()[np.flatnonzero(b)]
            hr_mean = arr_no_01.mean()
            hr_std = arr_no_01.std()
            b = np.where(b == 0, hr_mean, b)
            b = (b - hr_mean) / hr_std


            a = torch.from_numpy(np.expand_dims(a, 0))
            b = torch.from_numpy(np.expand_dims(b, 0))

            return a, b
        except RuntimeWarning:
            print("error!")




    def __len__(self):
        return len(self.image_filenames)


def get_dataloader(my_dataset, my_batch_size):
    torch.utils.data.DataLoader(my_dataset, batch_size=my_batch_size,
                          shuffle=False, num_workers=0, drop_last=True)