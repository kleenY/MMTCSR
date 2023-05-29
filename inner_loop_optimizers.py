import logging
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LSLRGradientDescentLearningRule(nn.Module):


    def __init__(self, device, total_num_inner_loop_steps, use_learnable_learning_rates, init_learning_rate=1e-3):

        super(LSLRGradientDescentLearningRule, self).__init__()
        print(init_learning_rate)
        # assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。
        assert init_learning_rate > 0., 'learning_rate should be positive.'

        self.init_learning_rate = torch.ones(1) * init_learning_rate   # torch.Size([1])
        self.init_learning_rate.to(device)
        self.total_num_inner_loop_steps = total_num_inner_loop_steps
        self.use_learnable_learning_rates = use_learnable_learning_rates


    def initialise(self, names_weights_dict):
        self.names_learning_rates_dict = nn.ParameterDict()
        for idx, (key, param) in enumerate(names_weights_dict.items()):
            self.names_learning_rates_dict[key.replace(".", "-")] = nn.Parameter(
                data=torch.ones(self.total_num_inner_loop_steps + 1) * self.init_learning_rate,
                requires_grad=self.use_learnable_learning_rates)

    def reset(self):


        pass

    def update_params(self, names_weights_dict, names_grads_wrt_params_dict, num_step, tau=0.1):

        return {
            key: names_weights_dict[key]
            - self.names_learning_rates_dict[key.replace(".", "-")][num_step]
            * names_grads_wrt_params_dict[key]
            for key in names_grads_wrt_params_dict.keys()
        }

