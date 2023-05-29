# -*- coding: utf-8 -*
# @Time : 2022/3/23 16:57
# @Author : 杨坤林
# @File : experiment_builder.py
# @Software : PyCharm

import tqdm
import os
import numpy as np
import sys
from utils.storage import build_experiment_folder, save_statistics, save_to_json
import time
import torch
from skimage.metrics import peak_signal_noise_ratio as sk_psnr

class ExperimentBuilder(object):
    def __init__(self, args, data, model, device):

        self.args, self.device = args, device

        self.model = model

        self.saved_models_filepath, self.logs_filepath, self.samples_filepath = build_experiment_folder(
            experiment_name=self.args.experiment_name)

        self.total_losses = {}

        self.state = {'best_val_psnr': 0.0, 'best_val_iter': 0, 'current_iter': 0}
        self.start_epoch = 0

        self.max_models_to_save = self.args.max_models_to_save
        self.create_summary_csv = False

        if self.args.continue_from_epoch == 'from_scratch':
            self.create_summary_csv = True

        elif self.args.continue_from_epoch == 'latest':   # 按照路径，将上一次的模型加载
            checkpoint = os.path.join(self.saved_models_filepath, "train_model_latest")
            print("attempting to find existing checkpoint", )
            if os.path.exists(checkpoint):
                self.state = \
                    self.model.load_model(model_save_dir=self.saved_models_filepath, model_name="train_model",
                                          model_idx='latest')
                self.start_epoch = int(self.state['current_iter'] / self.args.total_iter_per_epoch)

            else:
                self.args.continue_from_epoch = 'from_scratch'
                self.create_summary_csv = True
        elif int(self.args.continue_from_epoch) >= 0:
            self.state = \
                self.model.load_model(model_save_dir=self.saved_models_filepath, model_name="train_model",
                                      model_idx=self.args.continue_from_epoch)
            self.start_epoch = int(self.state['current_iter'] / self.args.total_iter_per_epoch)


        self.data = data(args=args, current_iter=self.state['current_iter'])

        print("train_seed {}, val_seed: {}, at start time".format(self.data.dataset.seed["train"],
                                                                  self.data.dataset.seed["val"]))
        self.total_epochs_before_pause = self.args.total_epochs_before_pause
        self.state['best_epoch'] = int(self.state['best_val_iter'] / self.args.total_iter_per_epoch)

        self.augment_flag = False
        self.start_time = time.time()
        self.epochs_done_in_this_run = 0
        print(self.state['current_iter'], int(self.args.total_iter_per_epoch * self.args.total_epochs))

    def build_summary_dict(self, total_losses, phase, summary_losses=None):

        if summary_losses is None:
            summary_losses = {}

        for key in total_losses:
            summary_losses["{}_{}_mean".format(phase, key)] = np.nanmean(total_losses[key])
            summary_losses["{}_{}_std".format(phase, key)] = np.nanstd(total_losses[key])

        return summary_losses

    def build_loss_summary_string(self, summary_losses):

        output_update = ""
        for key, value in zip(list(summary_losses.keys()), list(summary_losses.values())):
            if "loss" in key or "accuracy" in key:
                value = float(value)
                output_update += "{}: {:.4f}, ".format(key, value)

        return output_update

    def merge_two_dicts(self, first_dict, second_dict):

        z = first_dict.copy()
        z.update(second_dict)
        return z

    def train_iteration(self, train_sample, sample_idx, epoch_idx, total_losses, current_iter, pbar_train):

        x_support_set, x_target_set, y_support_set, y_target_set, seed = train_sample
        data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

        if sample_idx == 0:
            print("shape of data", x_support_set.shape, x_target_set.shape, y_support_set.shape,
                  y_target_set.shape)

        losses, _ = self.model.run_train_iter(data_batch=data_batch, epoch=epoch_idx)

        for key, value in zip(list(losses.keys()), list(losses.values())):
            if key not in total_losses:
                total_losses[key] = [float(value)]
            else:
                total_losses[key].append(float(value))

        train_losses = self.build_summary_dict(total_losses=total_losses, phase="train")
        train_output_update = self.build_loss_summary_string(losses)

        pbar_train.update(1)
        pbar_train.set_description("training phase {} -> {}".format(self.epoch, train_output_update))

        current_iter += 1

        return train_losses, total_losses, current_iter

    def evaluation_iteration(self, val_sample, total_losses, pbar_val, phase):

        x_support_set, x_target_set, y_support_set, y_target_set, seed = val_sample
        data_batch = (
            x_support_set, x_target_set, y_support_set, y_target_set)

        losses, _ = self.model.run_validation_iter(data_batch=data_batch)
        for key, value in zip(list(losses.keys()), list(losses.values())):
            if key not in total_losses:
                total_losses[key] = [float(value)]
            else:
                total_losses[key].append(float(value))

        val_losses = self.build_summary_dict(total_losses=total_losses, phase=phase)
        val_output_update = self.build_loss_summary_string(losses)

        pbar_val.update(1)
        pbar_val.set_description(
            "val_phase {} -> {}".format(self.epoch, val_output_update))

        return val_losses, total_losses

    def test_evaluation_iteration(self, val_sample, model_idx, sample_idx, per_model_per_batch_preds, pbar_test):

        x_support_set, x_target_set, y_support_set, y_target_set, seed = val_sample
        data_batch = (
            x_support_set, x_target_set, y_support_set, y_target_set)

        losses, per_task_preds = self.model.run_validation_iter(data_batch=data_batch)

        per_model_per_batch_preds[model_idx].extend(list(per_task_preds))

        test_output_update = self.build_loss_summary_string(losses)

        pbar_test.update(1)
        pbar_test.set_description(
            "test_phase {} -> {}".format(self.epoch, test_output_update))

        return per_model_per_batch_preds

    def save_models(self, model, epoch, state):


        model.save_model(model_save_dir=os.path.join(self.saved_models_filepath, "train_model_latest"),
                         state=state)

        print("saved models to", self.saved_models_filepath)

    def save_models1(self, model, epoch, state):

        model.save_model(model_save_dir=os.path.join(self.saved_models_filepath, "train_model_{}".format(int(epoch))),
                         state=state)

    def pack_and_save_metrics(self, start_time, create_summary_csv, train_losses, val_losses, state):

        epoch_summary_losses = self.merge_two_dicts(first_dict=train_losses, second_dict=val_losses)

        if 'per_epoch_statistics' not in state:
            state['per_epoch_statistics'] = {}

        for key, value in epoch_summary_losses.items():

            if key not in state['per_epoch_statistics']:
                state['per_epoch_statistics'][key] = [value]
            else:
                state['per_epoch_statistics'][key].append(value)

        epoch_summary_string = self.build_loss_summary_string(epoch_summary_losses)
        epoch_summary_losses["epoch"] = self.epoch
        epoch_summary_losses['epoch_run_time'] = time.time() - start_time

        if create_summary_csv:
            self.summary_statistics_filepath = save_statistics(self.logs_filepath, list(epoch_summary_losses.keys()),
                                                               create=True)
            self.create_summary_csv = False

        start_time = time.time()
        print("epoch {} -> {}".format(epoch_summary_losses["epoch"], epoch_summary_string))

        self.summary_statistics_filepath = save_statistics(self.logs_filepath,
                                                           list(epoch_summary_losses.values()))
        return start_time, state


    def evaluated_test_set_using_the_best_models(self, top_n_models):
        os.makedirs("./test_result/truth/")
        os.makedirs("./test_result/input/")
        os.makedirs("./test_result/pred/")
        per_epoch_statistics = self.state['per_epoch_statistics']
        val_psnr = np.copy(per_epoch_statistics['val_psnr_mean'])
        val_idx = np.array([i for i in range(len(val_psnr))])

        sorted_idx = np.argsort(val_psnr, axis=0).astype(dtype=np.int32)[::-1][:top_n_models]

        sorted_val_psnr = val_psnr[sorted_idx]
        val_idx = val_idx[sorted_idx]
        print(sorted_idx)
        print(sorted_val_psnr)

        top_n_idx = val_idx[:top_n_models]
        per_model_per_batch_preds = [[] for i in range(top_n_models)]  # 预测
        per_model_per_batch_targets = [[] for i in range(top_n_models)]   # 真实值
        per_model_per_batch_inputs = [[] for i in range(top_n_models)]  # 输入
        test_losses = [dict() for i in range(top_n_models)]
        for idx, model_idx in enumerate(top_n_idx):
            self.state = \
                self.model.load_model(model_save_dir=self.saved_models_filepath, model_name="train_model",
                                      model_idx=model_idx + 1)
            with tqdm.tqdm(total=int(self.args.num_evaluation_tasks / self.args.batch_size)) as pbar_test:
                for sample_idx, test_sample in enumerate(
                        self.data.get_test_batches(total_batches=int(self.args.num_evaluation_tasks / self.args.batch_size),
                                                   augment_images=False)):
                    #test_sample[3]，说明就是取得query的lable值
                    per_model_per_batch_targets[idx].extend(np.array(test_sample[3]))
                    per_model_per_batch_inputs[idx].extend(np.array(test_sample[1]))
                    per_model_per_batch_preds = self.test_evaluation_iteration(val_sample=test_sample,
                                                                               sample_idx=sample_idx,
                                                                               model_idx=idx,
                                                                               per_model_per_batch_preds=per_model_per_batch_preds,
                                                                               pbar_test=pbar_test)

        batch_num = int(self.args.num_evaluation_tasks / self.args.batch_size)

        test_psnr_result = [[0. for col in range(batch_num*self.args.batch_size)] for row in range(len(top_n_idx))]
        for modle_id in range(len(top_n_idx)):
            for batch_id in range(batch_num):
                for sample_batch in range(self.args.batch_size):
                    sr = per_model_per_batch_preds[modle_id][batch_id]
                    sr = sr[sample_batch].squeeze()

                    truth = per_model_per_batch_targets[modle_id][batch_id]
                    input = per_model_per_batch_inputs[modle_id][batch_id]

                    truth = truth[sample_batch].squeeze()
                    input = input[sample_batch].squeeze()
                    mask_of_truth = truth > 0
                    sr = sr * mask_of_truth
                    sr[np.isnan(sr)] = 0
                    truth[np.isnan(truth)] = 0
                    sr = self.my_normal(sr)
                    truth = self.my_normal(truth)

                    psnr = sk_psnr(truth, sr)

                    np.save("./test_result/truth/%d_%d_%d_truth.npy" % (modle_id, batch_id, sample_batch), truth)
                    np.save("./test_result/pred/%d_%d_%d_sr.npy" % (modle_id, batch_id, sample_batch), sr)
                    np.save("./test_result/input/%d_%d_%d_lr.npy" % (modle_id, batch_id, sample_batch), input)
                    test_psnr_result[modle_id][batch_id*self.args.batch_size+sample_batch] = psnr

        test_psnr_result = np.array(test_psnr_result)
        np.savetxt('out.txt', test_psnr_result, fmt="%f", delimiter=',')
        np.savetxt('out1.txt', top_n_idx, fmt="%d")
        mean_psnr_per_modle = np.nanmean(test_psnr_result, axis=1)

        test_losses = {"test_psnr_mean": mean_psnr_per_modle}

        _ = save_statistics(self.logs_filepath,
                            list(test_losses.keys()),
                            create=True, filename="test_summary.csv")

        summary_statistics_filepath = save_statistics(self.logs_filepath,
                                                      list(test_losses.values()),
                                                      create=False, filename="test_summary.csv")
        print(test_losses)
        print("saved test performance at", summary_statistics_filepath)

    def my_normal(self, x):
        smax = np.max(x)
        smin = np.min(x)
        s = (x - smin)/(smax - smin)
        return s

    def run_experiment(self):

        with tqdm.tqdm(initial=self.state['current_iter'],
                           total=int(self.args.total_iter_per_epoch * self.args.total_epochs)) as pbar_train:

            while (self.state['current_iter'] < (self.args.total_epochs * self.args.total_iter_per_epoch)) and (self.args.evaluate_on_test_set_only == False):

                for train_sample_idx, train_sample in enumerate(
                        self.data.get_train_batches(total_batches=int(self.args.total_iter_per_epoch *
                                                                      self.args.total_epochs) - self.state[
                                                                      'current_iter'],  # 这里的current是为了那些中途停止后继续训练的
                                                    augment_images=self.augment_flag)):

                    train_losses, total_losses, self.state['current_iter'] = self.train_iteration(
                        train_sample=train_sample,
                        total_losses=self.total_losses,
                        epoch_idx=(self.state['current_iter'] /
                                   self.args.total_iter_per_epoch),
                        pbar_train=pbar_train,
                        current_iter=self.state['current_iter'],
                        sample_idx=self.state['current_iter'])


                    if self.state['current_iter'] % self.args.total_iter_per_epoch == 0:

                        total_losses = {}
                        val_losses = {}

                        with tqdm.tqdm(total=int(self.args.num_evaluation_tasks / self.args.batch_size)) as pbar_val:

                            for _, val_sample in enumerate(
                                    self.data.get_val_batches(total_batches=int(self.args.num_evaluation_tasks / self.args.batch_size),
                                                              augment_images=False)):
                                val_losses, total_losses = self.evaluation_iteration(val_sample=val_sample,
                                                                                     total_losses=total_losses,
                                                                                     pbar_val=pbar_val, phase='val')

                            if val_losses["val_psnr_mean"] > self.state['best_val_psnr']:
                                print("Best validation psnr", val_losses["val_psnr_mean"])
                                self.state['best_val_psnr'] = val_losses["val_psnr_mean"]
                                self.state['best_val_iter'] = self.state['current_iter']
                                self.state['best_epoch'] = int(
                                    self.state['best_val_iter'] / self.args.total_iter_per_epoch)


                        self.epoch += 1
                        self.state = self.merge_two_dicts(first_dict=self.merge_two_dicts(first_dict=self.state,
                                                                                          second_dict=train_losses),
                                                          second_dict=val_losses)

                        if self.epoch % 50 == 0:

                            self.save_models1(model=self.model, epoch=self.epoch, state=self.state)

                        self.save_models(model=self.model, epoch=self.epoch, state=self.state)

                        self.start_time, self.state = self.pack_and_save_metrics(start_time=self.start_time,
                                                                                 create_summary_csv=self.create_summary_csv,
                                                                                 train_losses=train_losses,
                                                                                 val_losses=val_losses,
                                                                                 state=self.state)

                        self.total_losses = {}

                        self.epochs_done_in_this_run += 1

                        save_to_json(filename=os.path.join(self.logs_filepath, "summary_statistics.json"),
                                     dict_to_store=self.state['per_epoch_statistics'])

                        if self.epochs_done_in_this_run >= self.total_epochs_before_pause:
                            print("train_seed {}, val_seed: {}, at pause time".format(self.data.dataset.seed["train"],
                                                                                      self.data.dataset.seed["val"]))
                            sys.exit()

            self.evaluated_test_set_using_the_best_models(top_n_models=5)

