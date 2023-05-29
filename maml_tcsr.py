# -*- coding: utf-8 -*
# @Time : 2022/3/15 12:11
# @Author : 杨坤林
# @File : maml_tcsr.py
# @Software : PyCharm

'''
outer loop
'''


import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.optim.lr_scheduler import CosineAnnealingLR

import meta_tcsr_architecture as inner_modle

from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import normalized_root_mse as compare_nrmse
from skimage.metrics import peak_signal_noise_ratio as sk_psnr

from inner_loop_optimizers import LSLRGradientDescentLearningRule
from loss import VGGLoss
from discriminator import discriminator_for_vgg

def set_torch_seed(seed):   # 根据seed设置torch的随机数种子

    rng = np.random.RandomState(seed=seed)
    torch_seed = rng.randint(0, 999999)
    torch.manual_seed(seed=torch_seed)

    return rng

class MAML_TCSR(nn.Module):
    scheduler: CosineAnnealingLR

    def __init__(self, im_shape, device, args):

        super(MAML_TCSR, self).__init__()
        self.args = args
        self.device = device
        self.batch_size = args.batch_size
        self.use_cuda = args.use_cuda
        self.im_shape = im_shape  # 注释显示的是 batch, c, h, w
        self.current_epoch = 0

        self.rng = set_torch_seed(seed=args.seed)  # 上面的函数，设置torch随机数

        self.sr_modle = inner_modle.RDN(args).to(device=self.device)
        self.discriminator = discriminator_for_vgg(image_size=args.image_size)



        self.content_criterion = VGGLoss().to(device=self.device)
        self.adversarial_criterion = nn.BCEWithLogitsLoss().to(device=self.device)
        self.pixel_criterion = nn.L1Loss().to(device=self.device)
        self.task_learning_rate = args.task_learning_rate  # 任务学习率

        self.inner_loop_optimizer = LSLRGradientDescentLearningRule(device=device,
                                                                    init_learning_rate=self.task_learning_rate,
                                                                    total_num_inner_loop_steps=self.args.number_of_training_steps_per_iter,
                                                                    use_learnable_learning_rates=self.args.learnable_per_layer_per_step_inner_loop_learning_rate)

        self.inner_loop_optimizer.initialise(
            names_weights_dict=self.get_inner_loop_parameter_dict(params=self.sr_modle.named_parameters()))


        print("Inner Loop parameters")
        for key, value in self.inner_loop_optimizer.named_parameters():
            print(key, value.shape)

        self.use_cuda = args.use_cuda
        self.device = device
        self.args = args
        self.to(device)
        print("Outer Loop parameters")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.shape, param.device, param.requires_grad)


        self.optimizer = optim.Adam(self.trainable_parameters(), lr=args.meta_learning_rate, amsgrad=False)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.args.total_epochs,
                                                              eta_min=self.args.min_learning_rate)

        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), args.gan_lr, (0.9, 0.999))
        self.discriminator_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.discriminator_optimizer, T_max=self.args.total_epochs,
                                                              eta_min=self.args.min_learning_rate)

        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.to(torch.cuda.current_device())
                self.sr_modle = nn.DataParallel(module=self.sr_modle)
                self.discriminator = nn.DataParallel(module=self.discriminator)
            else:
                self.to(torch.cuda.current_device())

            self.device = torch.cuda.current_device()


    def get_per_step_loss_importance_vector(self):

        loss_weights = np.ones(shape=(self.args.number_of_training_steps_per_iter)) * (
                1.0 / self.args.number_of_training_steps_per_iter)
        decay_rate = 1.0 / self.args.number_of_training_steps_per_iter / self.args.multi_step_loss_num_epochs
        min_value_for_non_final_losses = 0.03 / self.args.number_of_training_steps_per_iter
        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(loss_weights[i] - (self.current_epoch * decay_rate), min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        curr_value = np.minimum(
            loss_weights[-1] + (self.current_epoch * (self.args.number_of_training_steps_per_iter - 1) * decay_rate),
            1.0 - ((self.args.number_of_training_steps_per_iter - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        loss_weights = torch.Tensor(loss_weights).to(device=self.device)
        return loss_weights

    def get_inner_loop_parameter_dict(self, params):

        return {
            name: param.to(device=self.device)
            for name, param in params
            if param.requires_grad
            and (
                not self.args.enable_inner_loop_optimizable_bn_params
                and "norm_layer" not in name
                or self.args.enable_inner_loop_optimizable_bn_params
            )
        }

    def apply_inner_loop_update(self, loss, names_weights_copy, use_second_order, current_step_idx):

        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            self.sr_modle.module.zero_grad(names_weights_copy)
        else:
            self.sr_modle.zero_grad(names_weights_copy)

        grads = torch.autograd.grad(loss, names_weights_copy.values(),
                                    create_graph=use_second_order, allow_unused=True)
        names_grads_copy = dict(zip(names_weights_copy.keys(), grads))

        names_weights_copy = {key: value[0] for key, value in names_weights_copy.items()}

        for key, grad in names_grads_copy.items():
            if grad is None:
                print('Grads not found for inner loop parameter', key)
            # names_grads_copy[key] = names_grads_copy[key].sum(dim=0)

        names_weights_copy = self.inner_loop_optimizer.update_params(names_weights_dict=names_weights_copy,
                                                                     names_grads_wrt_params_dict=names_grads_copy,
                                                                     num_step=current_step_idx)

        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        names_weights_copy = {
            name.replace('module.', ''): value.unsqueeze(0).repeat(
                [num_devices] + [1 for i in range(len(value.shape))]) for
            name, value in names_weights_copy.items()}

        return names_weights_copy



    # 不同任务的损失，应该需要修改，毕竟没有accuracies，可以考虑改为psnr和其他参数（nrmse）
    def get_across_task_loss_metrics(self, total_losses, total_psnr):
        losses = {'loss': torch.mean(torch.stack(total_losses))}

        losses['psnr'] = np.mean(total_psnr)

        return losses

    def forward(self, data_batch, epoch, use_second_order, use_multi_step_loss_optimization, num_steps, training_phase):

        # 外循环一批次的数据
        x_support_set, x_target_set, y_support_set, y_target_set = data_batch
        # 分x和y的原因，是输入和输出，support 和 target应该就是训练和测试
        # batch，samples ,这里尽可能后面，一个spc，当前准备设为64，到时候分8个epoch训练，直接用[-1,8],view一下就可以了


        total_losses = []

        # 注意是每次任务，所以是二维列表
        per_task_target_preds = [[] for i in range(len(x_target_set))]  # 用来存每次任务网络预测的输出值的列表
        self.sr_modle.zero_grad()  # 初始化梯度

        total_psnr = []


        # 这里相当于把买个batch单独拿出来了，相当于单个任务
        for task_id, (x_support_set_task, y_support_set_task, x_target_set_task, y_target_set_task) in enumerate(zip(x_support_set,
                              y_support_set,
                              x_target_set,
                              y_target_set)):
            task_losses = []
            content_losses = []
            pixel_losses = []
            per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector()  # 用来计算损失的向量，衡量每步损失的重要性

            names_weights_copy = self.get_inner_loop_parameter_dict(self.sr_modle.named_parameters())  # 返回内循环参数字典


            num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

            names_weights_copy = {
                name.replace('module.', ''): value.unsqueeze(0).repeat(
                    [num_devices] + [1 for i in range(len(value.shape))]) for
                name, value in names_weights_copy.items()}


            s, c, h, w = x_target_set_task.shape

            x_support_set_task = x_support_set_task.view(-1, c, h, w)   # A data batch of shape b, c, h, w
            y_support_set_task = y_support_set_task.view(-1, 1, h*2, w*2)   # 所以y就是lable
            x_target_set_task = x_target_set_task.view(-1, c, h, w)
            y_target_set_task = y_target_set_task.view(-1,1, h*2, w*2)


            my_target_loss_1, target_preds= self.net_forward(x=x_target_set_task,
                                                               y=y_target_set_task, weights=names_weights_copy,
                                                               backup_running_statistics=False, training=True,
                                                               num_step=0)
            my_target_loss_2, _= self.net_forward_2(x=x_target_set_task,
                                                               y=y_target_set_task, weights=names_weights_copy,
                                                               backup_running_statistics=False, training=True,
                                                               num_step=0)
            my_target_loss_3, _= self.net_forward(x=x_support_set_task,
                                                                   y=y_support_set_task, weights=names_weights_copy,
                                                                   backup_running_statistics=False, training=True,
                                                                   num_step=0)
            my_target_loss_4, _= self.net_forward_2(x=x_support_set_task,
                                                                   y=y_support_set_task, weights=names_weights_copy,
                                                                   backup_running_statistics=False, training=True,
                                                                   num_step=0)

            content_losses.append(my_target_loss_1)
            pixel_losses.append(my_target_loss_2)
            content_losses.append(my_target_loss_3)
            pixel_losses.append(my_target_loss_4)



            for num_step in range(num_steps):
                # 进行一个batch的inner loop的前向传播,这里的参数传递很可能出问题
                support_loss, support_preds = self.net_forward(
                    x=x_support_set_task,
                    y=y_support_set_task,
                    weights=names_weights_copy,  # 传递参数，重中之重，否则就跟元学习无关
                    backup_running_statistics=num_step == 0,  # 指示是否在运行后将批处理规范运行统计信息重置为其先前值的标志，只为验证集使用
                    training=True,
                    num_step=num_step,
                )


                names_weights_copy = self.apply_inner_loop_update(loss=support_loss,
                                                                  names_weights_copy=names_weights_copy,
                                                                  use_second_order=use_second_order,
                                                                  current_step_idx=num_step)

                names_weights_copy = {key: value[0] for key, value in names_weights_copy.items()}

                if use_multi_step_loss_optimization and training_phase and epoch < self.args.multi_step_loss_num_epochs:
                    target_loss, _ = self.net_forward(x=x_target_set_task,
                                                                 y=y_target_set_task,
                                                                 weights=names_weights_copy,
                                                                 backup_running_statistics=False,
                                                                 training=True,
                                                                 num_step=num_step)

                    task_losses.append(per_step_loss_importance_vectors[num_step] * target_loss)
                elif num_step == (self.args.number_of_training_steps_per_iter - 1):
                    target_loss, _ = self.net_forward(x=x_target_set_task,
                                                                 y=y_target_set_task, weights=names_weights_copy,
                                                                 backup_running_statistics=False, training=True,
                                                                 num_step=num_step)
                    task_losses.append(target_loss)


            per_task_target_preds[task_id] = target_preds.detach().cpu().numpy()
            per_task_target_truth = y_target_set_task.detach().cpu().numpy()

            final_target_pred = target_preds.detach().cpu().numpy()

            final_target_pred_normal = self.my_normal(final_target_pred)
            per_task_target_truth_normal = self.my_normal(per_task_target_truth)
            task_psnr = sk_psnr(final_target_pred_normal, per_task_target_truth_normal)

            task_losses = torch.sum(torch.stack(task_losses))
            content_losses = torch.sum(torch.stack(content_losses))
            pixel_losses = torch.sum(torch.stack(pixel_losses))
            total_losses.append(content_losses*0.2+task_losses*0.2+pixel_losses*0.6)

            total_psnr.append(task_psnr)


        # 计算总的不同任务的交叉损失
        losses = self.get_across_task_loss_metrics(total_losses=total_losses,
                                                   total_psnr=total_psnr)
        # 把每步的损失重要性向量加入损失列表中
        for idx, item in enumerate(per_step_loss_importance_vectors):
            losses['loss_importance_vector_{}'.format(idx)] = item.detach().cpu().numpy()

        return losses, per_task_target_preds



    def old_forward(self, data_batch, epoch, use_second_order, use_multi_step_loss_optimization, num_steps, training_phase):

        x_support_set, x_target_set, y_support_set, y_target_set = data_batch


        total_losses = []

        per_task_target_preds = [[] for i in range(len(x_target_set))]  # 用来存每次任务网络预测的输出值的列表
        self.sr_modle.zero_grad()  # 初始化梯度
        # task_accuracies = []
        total_psnr = []


        for task_id, (x_support_set_task, y_support_set_task, x_target_set_task, y_target_set_task) in enumerate(zip(x_support_set,
                              y_support_set,
                              x_target_set,
                              y_target_set)):
            task_losses = []

            per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector()  # 用来计算损失的向量，衡量每步损失的重要性

            names_weights_copy = self.get_inner_loop_parameter_dict(self.sr_modle.named_parameters())  # 返回内循环参数字典


            num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

            names_weights_copy = {
                name.replace('module.', ''): value.unsqueeze(0).repeat(
                    [num_devices] + [1 for i in range(len(value.shape))]) for
                name, value in names_weights_copy.items()}

            s, c, h, w = x_target_set_task.shape

            x_support_set_task = x_support_set_task.view(-1, c, h, w)   # A data batch of shape b, c, h, w
            y_support_set_task = y_support_set_task.view(-1, c, h*2, w*2)   # 所以y就是lable
            x_target_set_task = x_target_set_task.view(-1, c, h, w)
            y_target_set_task = y_target_set_task.view(-1,c, h*2, w*2)


            for num_step in range(num_steps):
                # 进行一个batch的inner loop的前向传播,这里的参数传递很可能出问题
                support_loss, support_preds = self.net_forward(
                    x=x_support_set_task,
                    y=y_support_set_task,
                    weights=names_weights_copy,
                    backup_running_statistics=num_step == 0,
                    training=True,
                    num_step=num_step,
                )


                names_weights_copy = self.apply_inner_loop_update(loss=support_loss,
                                                                  names_weights_copy=names_weights_copy,
                                                                  use_second_order=use_second_order,
                                                                  current_step_idx=num_step)

                names_weights_copy = {key: value[0] for key, value in names_weights_copy.items()}

                if use_multi_step_loss_optimization and training_phase and epoch < self.args.multi_step_loss_num_epochs:
                    target_loss, target_preds = self.net_forward(x=x_target_set_task,
                                                                 y=y_target_set_task,
                                                                 weights=names_weights_copy,
                                                                 backup_running_statistics=False,
                                                                 training=True,
                                                                 num_step=num_step)

                    task_losses.append(per_step_loss_importance_vectors[num_step] * target_loss)
                elif num_step == (self.args.number_of_training_steps_per_iter - 1):
                    target_loss, target_preds = self.net_forward(x=x_target_set_task,
                                                                 y=y_target_set_task, weights=names_weights_copy,
                                                                 backup_running_statistics=False, training=True,
                                                                 num_step=num_step)
                    task_losses.append(target_loss)


            per_task_target_preds[task_id] = target_preds.detach().cpu().numpy()
            per_task_target_truth = y_target_set_task.detach().cpu().numpy()

            final_target_pred = target_preds.detach().cpu().numpy()

            final_target_pred_normal = self.my_normal(final_target_pred)
            per_task_target_truth_normal = self.my_normal(per_task_target_truth)
            task_psnr = sk_psnr(final_target_pred_normal, per_task_target_truth_normal)

            task_losses = torch.sum(torch.stack(task_losses))

            total_losses.append(task_losses)

            total_psnr.append(task_psnr)

        losses = self.get_across_task_loss_metrics(total_losses=total_losses,
                                                   total_psnr=total_psnr)

        for idx, item in enumerate(per_step_loss_importance_vectors):
            losses['loss_importance_vector_{}'.format(idx)] = item.detach().cpu().numpy()

        return losses, per_task_target_preds

    # numpy数组归一化
    def my_normal(self, x):
        smax = np.max(x)
        smin = np.min(x)
        s = (x - smin)/(smax - smin)
        return s

    def net_forward(self, x, y, weights, backup_running_statistics, training, num_step):


        output = self.sr_modle(x = x, params = weights)


        new_output = torch.cat((output, output, output), 1)
        new_truth = torch.cat((y.detach(), y.detach(), y.detach()), 1)
        loss = self.content_criterion(new_output, new_truth)


        return loss, output

    def net_forward_2(self, x, y, weights, backup_running_statistics, training, num_step):
        # 这里传入的weights是33个网络框架的参数
        output = self.sr_modle(x=x, params=weights)
        output[output.isnan()] = torch.tensor(0.).cuda()
        output[output.isinf()] = torch.tensor(1.).cuda()

        loss = F.l1_loss(output, y)
        if loss.isnan().any():
            loss = torch.tensor(0.).cuda()
        # loss = self.content_criterion(output, y.detach())
        # loss = F.cross_entropy(input=preds, target=y)

        return loss, output

    def trainable_parameters(self):

        for param in self.parameters():
            if param.requires_grad:
                yield param

    def train_forward_prop(self, data_batch, epoch):

        losses, per_task_target_preds = self.forward(data_batch=data_batch, epoch=epoch,
                                                     use_second_order=self.args.second_order and
                                                                      epoch > self.args.first_order_to_second_order_epoch,
                                                     use_multi_step_loss_optimization=self.args.use_multi_step_loss_optimization,
                                                     num_steps=self.args.number_of_training_steps_per_iter,
                                                     training_phase=True)
        return losses, per_task_target_preds

    def evaluation_forward_prop(self, data_batch, epoch):

        losses, per_task_target_preds = self.forward(data_batch=data_batch, epoch=epoch, use_second_order=False,
                                                     use_multi_step_loss_optimization=True,
                                                     num_steps=self.args.number_of_evaluation_steps_per_iter,
                                                     training_phase=False)

        return losses, per_task_target_preds

    def meta_update(self, loss):

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

    def discriminator_meta_update(self, loss):

        self.discriminator_optimizer.zero_grad()
        loss.backward()

        self.discriminator_optimizer.step()

    def run_train_iter(self, data_batch, epoch):

        epoch = int(epoch)
        self.scheduler.step(epoch=epoch)  # 根据epoch调整学习率
        if self.current_epoch != epoch:
            self.current_epoch = epoch

        if self.training:
            self.train()  # Sets the module in training mode.

        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        x_support_set = torch.Tensor(x_support_set).float().to(device=self.device)
        x_target_set = torch.Tensor(x_target_set).float().to(device=self.device)
        y_support_set = torch.Tensor(y_support_set).float().to(device=self.device)
        y_target_set = torch.Tensor(y_target_set).float().to(device=self.device)

        data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)


        self.discriminator.train()
        x_target_set_task = x_target_set.view(-1, 3, 300, 300)
        y_target_set_task = y_target_set.view(-1, 1, 300*2, 300*2)
        batch_size = x_target_set_task.size(0)
        real_label = torch.full((batch_size, 1), 1, dtype=y_target_set_task.dtype).to(self.device)
        fake_label = torch.full((batch_size, 1), 0, dtype=y_target_set_task.dtype).to(self.device)
        self.discriminator.zero_grad()


        # 获得鉴别器损失
        names_weights_copy = self.get_inner_loop_parameter_dict(self.sr_modle.named_parameters())
        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        names_weights_copy = {
            name.replace('module.', ''): value.unsqueeze(0).repeat(
                [num_devices] + [1 for i in range(len(value.shape))]) for
            name, value in names_weights_copy.items()}
        first_real_output = self.sr_modle(x=x_target_set_task, params=names_weights_copy)
        fake_first_output = self.discriminator(x=first_real_output.detach())
        real_first_output = self.discriminator(x=y_target_set_task)
        # Adversarial loss for real and fake images (relativistic average GAN)
        d_loss_real = self.adversarial_criterion(real_first_output - torch.mean(fake_first_output), real_label)
        d_loss_fake = self.adversarial_criterion(fake_first_output - torch.mean(real_first_output), fake_label)
        # Count all discriminator losses.
        d_loss = (d_loss_real + d_loss_fake) / 2
        self.discriminator_meta_update(loss=d_loss)
        self.discriminator_scheduler.get_lr()
        self.discriminator_optimizer.zero_grad()

        # 更新生成器

        # 获得鉴别器损失
        second_real_output = self.sr_modle(x=x_target_set_task, params=names_weights_copy)
        fake_second_output = self.discriminator(x=second_real_output.detach())
        real_second_output = self.discriminator(x=y_target_set_task)
        # Adversarial loss for real and fake images (relativistic average GAN)
        adversarial_loss = self.adversarial_criterion(fake_second_output - torch.mean(real_second_output), real_label)

        # 外循环
        losses, per_task_target_preds = self.train_forward_prop(data_batch=data_batch, epoch=epoch)
        # 参数更新

        self.meta_update(loss=losses['loss']+adversarial_loss*0.1)

        losses['learning_rate'] = self.scheduler.get_lr()[0]
        self.optimizer.zero_grad()
        self.zero_grad()

        return losses, per_task_target_preds

    def run_validation_iter(self, data_batch):

        if not self.training:
            self.eval()

        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        x_support_set = torch.Tensor(x_support_set).float().to(device=self.device)
        x_target_set = torch.Tensor(x_target_set).float().to(device=self.device)
        y_support_set = torch.Tensor(y_support_set).float().to(device=self.device)
        y_target_set = torch.Tensor(y_target_set).float().to(device=self.device)

        data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

        losses, per_task_target_preds = self.evaluation_forward_prop(data_batch=data_batch, epoch=self.current_epoch)



        return losses, per_task_target_preds

    def save_model(self, model_save_dir, state):

        state['network'] = self.state_dict()
        state['optimizer'] = self.optimizer.state_dict()
        state['optimizer_D'] = self.discriminator_optimizer.state_dict()
        torch.save(state, f=model_save_dir)



    def load_model(self, model_save_dir, model_name, model_idx):

        filepath = os.path.join(model_save_dir, "{}_{}".format(model_name, model_idx))
        state = torch.load(filepath)
        state_dict_loaded = state['network']
        self.optimizer.load_state_dict(state['optimizer'])
        self.discriminator_optimizer.load_state_dict(state['optimizer_D'])
        self.load_state_dict(state_dict=state_dict_loaded)
        return state