import os
import numpy as np
import shutil

path_old = "/media/aita-ocean/data/YKL/new_meta_data"
path_new = "/media/aita-ocean/data/YKL/meta_train_data"


def data_split():
    print("start!")
    task_list = os.listdir(path_old)
    for task in task_list:
        i = 0
        index_list = os.listdir(os.path.join(path_old, task, "lr"))
        index_list.sort()
        trn_index = index_list[0:799]
        val_index = index_list[800:999]
        test_index = index_list[1000:1199]
        temp_lr_path = os.path.join(path_old, task, "lr")
        temp_hr_path = os.path.join(path_old, task, "hr")
        # 创建新的保存地址
        temp_trn_lr_save_path = os.path.join(path_new, "train", task, "lr")
        temp_trn_hr_save_path = os.path.join(path_new, "train", task, "hr")
        temp_val_lr_save_path = os.path.join(path_new, "val", task, "lr")
        temp_val_hr_save_path = os.path.join(path_new, "val", task, "hr")
        temp_test_lr_save_path = os.path.join(path_new, "test", task, "lr")
        temp_test_hr_save_path = os.path.join(path_new, "test", task, "hr")
        if not os.path.isdir(temp_trn_lr_save_path):  # 判断是否存在，不存在，创建
            os.makedirs(temp_trn_lr_save_path)
        if not os.path.isdir(temp_trn_hr_save_path):  # 判断是否存在，不存在，创建
            os.makedirs(temp_trn_hr_save_path)
        if not os.path.isdir(temp_val_lr_save_path):  # 判断是否存在，不存在，创建
            os.makedirs(temp_val_lr_save_path)
        if not os.path.isdir(temp_val_hr_save_path):  # 判断是否存在，不存在，创建
            os.makedirs(temp_val_hr_save_path)
        if not os.path.isdir(temp_test_lr_save_path):  # 判断是否存在，不存在，创建
            os.makedirs(temp_test_lr_save_path)
        if not os.path.isdir(temp_test_hr_save_path):  # 判断是否存在，不存在，创建
            os.makedirs(temp_test_hr_save_path)

        for trn_file in trn_index:
            shutil.copy(os.path.join(temp_lr_path, trn_file), os.path.join(temp_trn_lr_save_path, "{}.npy".format(i)))
            shutil.copy(os.path.join(temp_hr_path, trn_file), os.path.join(temp_trn_hr_save_path, "{}.npy".format(i)))
            i += 1
        for val_file in val_index:
            shutil.copy(os.path.join(temp_lr_path, val_file), os.path.join(temp_val_lr_save_path, "{}.npy".format(i)))
            shutil.copy(os.path.join(temp_hr_path, val_file), os.path.join(temp_val_hr_save_path, "{}.npy".format(i)))
            i += 1
        for test_file in test_index:
            shutil.copy(os.path.join(temp_lr_path, test_file), os.path.join(temp_test_lr_save_path, "{}.npy".format(i)))
            shutil.copy(os.path.join(temp_hr_path, test_file), os.path.join(temp_test_hr_save_path, "{}.npy".format(i)))
            i += 1
        print("over+1!")


if __name__ == "__main__":
    data_split()
