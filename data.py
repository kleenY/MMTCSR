# -*- coding: utf-8 -*
# @Time : 2022/3/18 17:53
# @Author : 杨坤林
# @File : data.py
# @Software : PyCharm

import json
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import tqdm
import concurrent.futures
import pickle
import torch
import random
from torchvision import transforms

from utils.parser_utils import get_args



# 垂直翻转
def flip_v(source, target, target3, target4):
    test_h = np.copy(source)
    test_h2 = np.copy(target)
    test_h3 = np.copy(target3)
    test_h4 = np.copy(target4)
    return test_h[::-1], test_h2[::-1], test_h3[::-1], test_h4[::-1]

# 旋转180度
def flip180(arr, target, target3, target4):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)

    new_arr2 = target.reshape(target.size)
    new_arr2 = new_arr2[::-1]
    new_arr2 = new_arr2.reshape(target.shape)

    new_arr3 = target3.reshape(target3.size)
    new_arr3 = new_arr3[::-1]
    new_arr3 = new_arr3.reshape(target3.shape)

    new_arr4 = target4.reshape(target4.size)
    new_arr4 = new_arr4[::-1]
    new_arr4 = new_arr4.reshape(target4.shape)

    return new_arr, new_arr2, new_arr3, new_arr4

# 水平翻转
def flip_h(arr, target, target3, target4):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    test_v = new_arr[::-1]

    new_arr2 = target.reshape(target.size)
    new_arr2 = new_arr2[::-1]
    new_arr2 = new_arr2.reshape(target.shape)
    test_v2 = new_arr2[::-1]

    new_arr3 = target3.reshape(target3.size)
    new_arr3 = new_arr3[::-1]
    new_arr3 = new_arr3.reshape(target3.shape)
    test_v3 = new_arr3[::-1]

    new_arr4 = target4.reshape(target4.size)
    new_arr4 = new_arr4[::-1]
    new_arr4 = new_arr4.reshape(target4.shape)
    test_v4 = new_arr4[::-1]

    return test_v, test_v2, test_v3, test_v4

# 旋转270度（逆时针旋转90度）

def flip90_left(arr, target, target3, target4):
    new_arr = np.transpose(arr)
    new_arr = new_arr[::-1]

    new_arr2 = np.transpose(target)
    new_arr2 = new_arr2[::-1]

    new_arr3 = np.transpose(target3)
    new_arr3 = new_arr3[::-1]

    new_arr4 = np.transpose(target4)
    new_arr4 = new_arr4[::-1]
    return new_arr, new_arr2, new_arr3, new_arr4

# 旋转90度
def flip90_right(arr, target, target3, target4):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    new_arr = np.transpose(new_arr)[::-1]

    new_arr2 = target.reshape(target.size)
    new_arr2 = new_arr2[::-1]
    new_arr2 = new_arr2.reshape(target.shape)
    new_arr2 = np.transpose(new_arr2)[::-1]

    new_arr3 = target3.reshape(target3.size)
    new_arr3 = new_arr3[::-1]
    new_arr3 = new_arr3.reshape(target3.shape)
    new_arr3 = np.transpose(new_arr3)[::-1]

    new_arr4 = target4.reshape(target4.size)
    new_arr4 = new_arr4[::-1]
    new_arr4 = new_arr4.reshape(target4.shape)
    new_arr4 = np.transpose(new_arr4)[::-1]

    return new_arr, new_arr2, new_arr3, new_arr4



class Numpy_Transform(object):
    def __init__(self, mode='train', probability=0.5):
        self.mode = mode
        self.probability = probability


    def __call__(self, sample, target, target3, target4):
        if self.mode == 'train':
            if round(np.random.uniform(0, 1), 1) <= self.probability:
                image1, image2, image3, image4 = sample, target, target3, target4
                image1, image2, image3, image4 = random.choice([flip_v,flip180, flip_h,
                                                                flip90_right, flip90_left])(image1, image2, image3, image4)
                return image1, image2, image3, image4
            else:
                return sample, target, target3, target4

        if self.mode == 'test' or self.mode == 'infer':
            return sample, target, target3, target4




class Meta_Tcsr_Dataset(Dataset):
    def __init__(self, args):

        self.data_path = args.dataset_path
        self.dataset_name = args.dataset_name
        self.data_loaded_in_memory = False
        self.image_height, self.image_width, self.image_channel = args.image_height, args.image_width, args.image_channels
        self.args = args


        self.current_set_name = "train"
        self.num_target_samples = args.num_target_samples  # 每次任务的query数量


        val_rng = np.random.RandomState(seed=args.val_seed)  # 随机数种子
        val_seed = val_rng.randint(1, 999999)
        train_rng = np.random.RandomState(seed=args.train_seed)
        train_seed = train_rng.randint(1, 999999)
        test_rng = np.random.RandomState(seed=args.val_seed)
        test_seed = test_rng.randint(1, 999999)
        args.val_seed = val_seed
        args.train_seed = train_seed
        args.test_seed = test_seed
        self.init_seed = {"train": args.train_seed, "val": args.val_seed, 'test': args.val_seed}
        self.seed = {"train": args.train_seed, "val": args.val_seed, 'test': args.val_seed}

        self.num_of_gpus = args.num_of_gpus
        self.batch_size = args.batch_size

        self.augment_images = False  # 数据集增强
        self.num_samples_per_class = args.num_samples_per_class  # 每个类别的样本数，相对于我们代码里面的support_set 的采样个数

        self.rng = np.random.RandomState(seed=self.seed['val'])

        # 数据集大小，理想数据集
        self.data_length = {"train":args.train_size*24, "val":args.val_size*24, "test":args.test_size*24}
        print("data", self.data_length)
        # 这里就相当于load dataset 了

        self.task_list = os.listdir(args.task_path)
        self.train_index_list = os.listdir(args.train_index_path)
        self.val_index_list = os.listdir(args.val_index_path)
        self.test_index_list = os.listdir(args.test_index_path)
        self.transform = Numpy_Transform(probability=0.5)






    def get_set(self, dataset_name, seed, augment_images=False):

        rng = np.random.RandomState(seed)  # 这个seed每次dataloader都不一样

        selected_task = rng.choice(self.task_list, replace=False)   # 随机选一个任务

        if dataset_name == 'train':
            selected_index = rng.choice(self.train_index_list,
                                        size=self.num_samples_per_class + self.num_target_samples, replace=False)
        elif dataset_name == 'val':
            selected_index = rng.choice(self.val_index_list,
                                        size=self.num_samples_per_class + self.num_target_samples, replace=False)
        else:
            selected_index = rng.choice(self.test_index_list,
                                        size=self.num_samples_per_class + self.num_target_samples, replace=False)


        x_lr = []
        y_hr = []
        z_matched = []


        for sample in selected_index:

            image_lr = np.load(os.path.join(self.data_path, dataset_name, selected_task, 'lr', sample))
            image_hr = np.load(os.path.join(self.data_path, dataset_name, selected_task, 'hr', sample))
            image_matched = np.load(os.path.join(self.data_path, dataset_name, selected_task, 'matched_lr', sample))
            image_matched2 = np.load(os.path.join(self.data_path, dataset_name, selected_task, 'matched_lr2', sample))

            if dataset_name == 'train':
                image_lr, image_hr, image_matched, image_matched2 = self.transform(image_lr, image_hr,
                                                                               image_matched, image_matched2)



            image_lr = (image_lr - 100) / (350 - 100)

            # # hr 归一化

            image_hr = (image_hr - 100) / (350 - 100)
            #
            # # matched 归一化

            image_matched = (image_matched - 100) / (350 - 100)
            image_matched2 = (image_matched2 - 100) / (350 - 100)

            # 转化为tensor格式

            a = torch.from_numpy(np.expand_dims(image_lr, 0))
            b = torch.from_numpy(np.expand_dims(image_hr, 0))
            c = torch.from_numpy(np.expand_dims(image_matched, 0))
            e = torch.from_numpy(np.expand_dims(image_matched2, 0))
            d = torch.concat([a, c, e], dim=0)

            d = d.float()

            x_lr.append(d)
            y_hr.append(b)


        x_lr = torch.stack(x_lr)
        y_hr = torch.stack(y_hr)

        support_set_lr = x_lr[:self.num_samples_per_class]    # 每次内循环的采样数
        support_set_hr = y_hr[:self.num_samples_per_class]

        target_set_lr = x_lr[self.num_samples_per_class:]
        target_set_hr = y_hr[self.num_samples_per_class:]


        return support_set_lr, target_set_lr, support_set_hr, target_set_hr, seed


    def __len__(self):
        return self.data_length[self.current_set_name]

    def length(self, set_name):
        self.switch_set(set_name=set_name)
        return len(self)

    def set_augmentation(self, augment_images):
        self.augment_images = augment_images

    def switch_set(self, set_name, current_iter=None):
        self.current_set_name = set_name
        if set_name == "train":
            self.update_seed(dataset_name=set_name, seed=self.init_seed[set_name] + current_iter)

    def update_seed(self, dataset_name, seed=100):
        self.seed[dataset_name] = seed

    def __getitem__(self, idx):
        support_set_images, target_set_image, support_set_labels, target_set_label, seed = \
            self.get_set(self.current_set_name, seed=self.seed[self.current_set_name] + idx,  # 产生种子用，train、val之类的
                         augment_images=self.augment_images)

        return support_set_images, target_set_image, support_set_labels, target_set_label, seed

    def reset_seed(self):
        self.seed = self.init_seed



class MetaLearningSystemDataLoader(object):
    def __init__(self, args, current_iter=0):

        self.num_of_gpus = args.num_of_gpus
        self.batch_size = args.batch_size
        self.samples_per_iter = args.samples_per_iter
        self.num_workers = args.num_dataprovider_workers
        self.total_train_iters_produced = 0
        self.dataset = Meta_Tcsr_Dataset(args=args)

        self.full_data_length = self.dataset.data_length
        self.continue_from_iter(current_iter=current_iter)
        self.args = args

    def get_dataloader(self):

        return DataLoader(self.dataset, batch_size=(self.num_of_gpus * self.batch_size * self.samples_per_iter),
                          shuffle=False, num_workers=self.num_workers, drop_last=True)

    def continue_from_iter(self, current_iter):

        self.total_train_iters_produced += (current_iter * (self.num_of_gpus * self.batch_size * self.samples_per_iter))

    def get_train_batches(self, total_batches=-1, augment_images=False):

        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:

            self.dataset.data_length["train"] = total_batches * self.dataset.batch_size

        self.dataset.switch_set(set_name="train", current_iter=self.total_train_iters_produced)

        self.total_train_iters_produced += (self.num_of_gpus * self.batch_size * self.samples_per_iter)
        for sample_id, sample_batched in enumerate(self.get_dataloader()):
            yield sample_batched


    def get_val_batches(self, total_batches=-1, augment_images=False):

        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length['val'] = total_batches * self.dataset.batch_size
        self.dataset.switch_set(set_name="val")

        for sample_id, sample_batched in enumerate(self.get_dataloader()):
            yield sample_batched


    def get_test_batches(self, total_batches=-1, augment_images=False):

        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length['test'] = total_batches * self.dataset.batch_size
        self.dataset.switch_set(set_name='test')

        for sample_id, sample_batched in enumerate(self.get_dataloader()):
            yield sample_batched

