# -*- coding: utf-8 -*
# @Time : 2022/3/15 12:14
# @Author : 杨坤林
# @File : meta_tcsr_architecture.py
# @Software : PyCharm

import numbers
from copy import copy

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
# from my_attention import SpatialAttention as sam

def extract_top_level_dict(current_dict):

    output_dict = {}

    for key in current_dict.keys():
        name = key.replace("layer_dict.", "")
        name = name.replace("layer_dict.", "")
        name = name.replace("block_dict.", "")
        name = name.replace("module-", "")
        top_level = name.split(".")[0]
        sub_level = ".".join(name.split(".")[1:])

        if top_level in output_dict:
            new_item = {key: value for key, value in output_dict[top_level].items()}
            new_item[sub_level] = current_dict[key]
            output_dict[top_level] = new_item

        elif sub_level == "":
            output_dict[top_level] = current_dict[key]
        else:
            output_dict[top_level] = {sub_level: current_dict[key]}
    #print(current_dict.keys(), output_dict.keys())
    return output_dict


class MetaConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_bias=True, groups=1, dilation_rate=1):

        super(MetaConv2dLayer, self).__init__()
        num_filters = out_channels
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation_rate = int(dilation_rate)
        self.use_bias = use_bias
        self.groups = int(groups)
        self.weight = nn.Parameter(torch.empty(num_filters, in_channels, kernel_size, kernel_size))
        nn.init.xavier_uniform_(self.weight)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_filters))

    def forward(self, x, params=None):

        if params is not None:
            if isinstance(params, dict):
                params = extract_top_level_dict(current_dict=params)
                if self.use_bias:
                    (weight, bias) = params["weight"], params["bias"]
                else:
                    (weight) = params["weight"]
                    bias = None
            else:
                (weight) = params
                bias = None
        elif self.use_bias:
            weight, bias = self.weight, self.bias
        else:
            weight = self.weight
            bias = None

        return F.conv2d(
            input=x,
            weight=weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation_rate,
            groups=self.groups
        )




class MetaLinearLayer(nn.Module):
    def __init__(self, input_shape, num_filters, use_bias):

        super(MetaLinearLayer, self).__init__()
        b, c = input_shape

        self.use_bias = use_bias
        self.weights = nn.Parameter(torch.ones(num_filters, c))
        nn.init.xavier_uniform_(self.weights)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_filters))

    def forward(self, x, params=None):

        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            if self.use_bias:
                (weight, bias) = params["weights"], params["bias"]
            else:
                (weight) = params["weights"]
                bias = None
        elif self.use_bias:
            weight, bias = self.weights, self.bias
        else:
            weight = self.weights
            bias = None
        return F.linear(input=x, weight=weight, bias=bias)


class MetaBatchNormLayer(nn.Module):
    def __init__(self, num_features, device, args, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, meta_batch_norm=True, no_learnable_params=False,
                 use_per_step_bn_statistics=False):

        super(MetaBatchNormLayer, self).__init__()
        self.num_features = num_features
        self.eps = eps

        self.affine = affine
        self.track_running_stats = track_running_stats
        self.meta_batch_norm = meta_batch_norm
        self.num_features = num_features
        self.device = device
        self.use_per_step_bn_statistics = use_per_step_bn_statistics
        self.args = args
        self.learnable_gamma = self.args.learnable_bn_gamma
        self.learnable_beta = self.args.learnable_bn_beta

        if use_per_step_bn_statistics:
            self.running_mean = nn.Parameter(torch.zeros(args.number_of_training_steps_per_iter, num_features),
                                             requires_grad=False)
            self.running_var = nn.Parameter(torch.ones(args.number_of_training_steps_per_iter, num_features),
                                            requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(args.number_of_training_steps_per_iter, num_features),
                                     requires_grad=self.learnable_beta)
            self.weight = nn.Parameter(torch.ones(args.number_of_training_steps_per_iter, num_features),
                                       requires_grad=self.learnable_gamma)
        else:
            self.running_mean = nn.Parameter(torch.zeros(num_features), requires_grad=False)
            self.running_var = nn.Parameter(torch.zeros(num_features), requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(num_features),
                                     requires_grad=self.learnable_beta)
            self.weight = nn.Parameter(torch.ones(num_features),
                                       requires_grad=self.learnable_gamma)

        if self.args.enable_inner_loop_optimizable_bn_params:
            self.bias = nn.Parameter(torch.zeros(num_features),
                                     requires_grad=self.learnable_beta)
            self.weight = nn.Parameter(torch.ones(num_features),
                                       requires_grad=self.learnable_gamma)

        self.backup_running_mean = torch.zeros(self.running_mean.shape)
        self.backup_running_var = torch.ones(self.running_var.shape)

        self.momentum = momentum

    def forward(self, input, num_step, params=None, training=False, backup_running_statistics=False):

        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            (weight, bias) = params["weight"], params["bias"]
            #print(num_step, params['weight'])
        else:
            #print(num_step, "no params")
            weight, bias = self.weight, self.bias

        if self.use_per_step_bn_statistics:
            running_mean = self.running_mean[num_step]
            running_var = self.running_var[num_step]
            if (
                params is None
                and not self.args.enable_inner_loop_optimizable_bn_params
            ):
                bias = self.bias[num_step]
                weight = self.weight[num_step]
        else:
            running_mean = None
            running_var = None


        if backup_running_statistics and self.use_per_step_bn_statistics:
            self.backup_running_mean.data = copy(self.running_mean.data)
            self.backup_running_var.data = copy(self.running_var.data)

        momentum = self.momentum

        return F.batch_norm(input, running_mean, running_var, weight, bias,
                              training=True, momentum=momentum, eps=self.eps)

    def restore_backup_stats(self):

        if self.use_per_step_bn_statistics:
            self.running_mean = nn.Parameter(self.backup_running_mean.to(device=self.device), requires_grad=False)
            self.running_var = nn.Parameter(self.backup_running_var.to(device=self.device), requires_grad=False)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

class MetaLayerNormLayer(nn.Module):
    def __init__(self, input_feature_shape, eps=1e-5, elementwise_affine=True):

        super(MetaLayerNormLayer, self).__init__()
        if isinstance(input_feature_shape, numbers.Integral):
            input_feature_shape = (input_feature_shape,)
        self.normalized_shape = torch.Size(input_feature_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*input_feature_shape), requires_grad=False)
            self.bias = nn.Parameter(torch.Tensor(*input_feature_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):

        if self.elementwise_affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

    def forward(self, input, num_step, params=None, training=False, backup_running_statistics=False):

        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            bias = params["bias"]
        else:
            bias = self.bias
            #print('no inner loop params', self)

        return F.layer_norm(
            input, self.normalized_shape, self.weight, bias, self.eps)

    def restore_backup_stats(self):
        pass

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)

class MetaConvNormLayerReLU(nn.Module):
    def __init__(self, input_shape, num_filters, kernel_size, stride, padding, use_bias, args, normalization=True,
                 meta_layer=True, no_bn_learnable_params=False, device=None):

        super(MetaConvNormLayerReLU, self).__init__()
        self.normalization = normalization
        self.use_per_step_bn_statistics = args.per_step_bn_statistics
        self.input_shape = input_shape
        self.args = args
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.meta_layer = meta_layer
        self.no_bn_learnable_params = no_bn_learnable_params
        self.device = device
        self.layer_dict = nn.ModuleDict()
        self.build_block()

    def build_block(self):

        x = torch.zeros(self.input_shape)

        out = x

        self.conv = MetaConv2dLayer(in_channels=out.shape[1], out_channels=self.num_filters,
                                    kernel_size=self.kernel_size,
                                    stride=self.stride, padding=self.padding, use_bias=self.use_bias)



        out = self.conv(out)

        if self.normalization:
            if self.args.norm_layer == "batch_norm":
                self.norm_layer = MetaBatchNormLayer(out.shape[1], track_running_stats=True,
                                                     meta_batch_norm=self.meta_layer,
                                                     no_learnable_params=self.no_bn_learnable_params,
                                                     device=self.device,
                                                     use_per_step_bn_statistics=self.use_per_step_bn_statistics,
                                                     args=self.args)
            elif self.args.norm_layer == "layer_norm":
                self.norm_layer = MetaLayerNormLayer(input_feature_shape=out.shape[1:])

            out = self.norm_layer(out, num_step=0)

        out = F.leaky_relu(out)

        print(out.shape)

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):

        batch_norm_params = None
        conv_params = None
        activation_function_pre_params = None

        if params is not None:
            params = extract_top_level_dict(current_dict=params)

            if self.normalization:
                if 'norm_layer' in params:
                    batch_norm_params = params['norm_layer']

                if 'activation_function_pre' in params:
                    activation_function_pre_params = params['activation_function_pre']

            conv_params = params['conv']

        out = x


        out = self.conv(out, params=conv_params)

        if self.normalization:
            out = self.norm_layer.forward(out, num_step=num_step,
                                          params=batch_norm_params, training=training,
                                          backup_running_statistics=backup_running_statistics)

        out = F.leaky_relu(out)

        return out

    def restore_backup_stats(self):

        if self.normalization:
            self.norm_layer.restore_backup_stats()


class MetaNormLayerConvReLU(nn.Module):
    def __init__(self, input_shape, num_filters, kernel_size, stride, padding, use_bias, args, normalization=True,
                 meta_layer=True, no_bn_learnable_params=False, device=None):

        super(MetaNormLayerConvReLU, self).__init__()
        self.normalization = normalization
        self.use_per_step_bn_statistics = args.per_step_bn_statistics
        self.input_shape = input_shape
        self.args = args
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.meta_layer = meta_layer
        self.no_bn_learnable_params = no_bn_learnable_params
        self.device = device
        self.layer_dict = nn.ModuleDict()
        self.build_block()

    def build_block(self):

        x = torch.zeros(self.input_shape)

        out = x
        if self.normalization:
            if self.args.norm_layer == "batch_norm":
                self.norm_layer = MetaBatchNormLayer(self.input_shape[1], track_running_stats=True,
                                                     meta_batch_norm=self.meta_layer,
                                                     no_learnable_params=self.no_bn_learnable_params,
                                                     device=self.device,
                                                     use_per_step_bn_statistics=self.use_per_step_bn_statistics,
                                                     args=self.args)
            elif self.args.norm_layer == "layer_norm":
                self.norm_layer = MetaLayerNormLayer(input_feature_shape=out.shape[1:])

            out = self.norm_layer.forward(out, num_step=0)
        self.conv = MetaConv2dLayer(in_channels=out.shape[1], out_channels=self.num_filters,
                                    kernel_size=self.kernel_size,
                                    stride=self.stride, padding=self.padding, use_bias=self.use_bias)


        self.layer_dict['activation_function_pre'] = nn.LeakyReLU()


        out = self.layer_dict['activation_function_pre'].forward(self.conv.forward(out))
        print(out.shape)

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):

        batch_norm_params = None

        if params is not None:
            params = extract_top_level_dict(current_dict=params)

            if self.normalization and 'norm_layer' in params:
                batch_norm_params = params['norm_layer']

            conv_params = params['conv']
        else:
            conv_params = None
            #print('no inner loop params', self)

        out = x

        if self.normalization:
            out = self.norm_layer.forward(out, num_step=num_step,
                                          params=batch_norm_params, training=training,
                                          backup_running_statistics=backup_running_statistics)

        out = self.conv.forward(out, params=conv_params)
        out = self.layer_dict['activation_function_pre'].forward(out)

        return out

    def restore_backup_stats(self):

        if self.normalization:
            self.norm_layer.restore_backup_stats()


class VGGReLUNormNetwork(nn.Module):
    def __init__(self, im_shape, num_output_classes, args, device, meta_classifier=True):

        super(VGGReLUNormNetwork, self).__init__()
        b, c, self.h, self.w = im_shape
        self.device = device
        self.total_layers = 0
        self.args = args
        self.upscale_shapes = []
        self.cnn_filters = args.cnn_num_filters
        self.input_shape = list(im_shape)
        self.num_stages = args.num_stages
        self.num_output_classes = num_output_classes

        if args.max_pooling:
            print("Using max pooling")
            self.conv_stride = 1
        else:
            print("Using strided convolutions")
            self.conv_stride = 2
        self.meta_classifier = meta_classifier

        self.build_network()
        print("meta network params")
        for name, param in self.named_parameters():
            print(name, param.shape)

    def build_network(self):

        x = torch.zeros(self.input_shape)
        out = x
        self.layer_dict = nn.ModuleDict()
        self.upscale_shapes.append(x.shape)

        for i in range(self.num_stages):
            self.layer_dict['conv{}'.format(i)] = MetaConvNormLayerReLU(input_shape=out.shape,
                                                                        num_filters=self.cnn_filters,
                                                                        kernel_size=3, stride=self.conv_stride,
                                                                        padding=self.args.conv_padding,
                                                                        use_bias=True, args=self.args,
                                                                        normalization=True,
                                                                        meta_layer=self.meta_classifier,
                                                                        no_bn_learnable_params=False,
                                                                        device=self.device)
            out = self.layer_dict['conv{}'.format(i)](out, training=True, num_step=0)

            if self.args.max_pooling:
                out = F.max_pool2d(input=out, kernel_size=(2, 2), stride=2, padding=0)


        if not self.args.max_pooling:
            out = F.avg_pool2d(out, out.shape[2])

        self.encoder_features_shape = list(out.shape)
        out = out.view(out.shape[0], -1)

        self.layer_dict['linear'] = MetaLinearLayer(input_shape=(out.shape[0], np.prod(out.shape[1:])),
                                                    num_filters=self.num_output_classes, use_bias=True)

        out = self.layer_dict['linear'](out)
        print("VGGNetwork build", out.shape)

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):

        param_dict = {}

        if params is not None:
            params = {key: value[0] for key, value in params.items()}
            param_dict = extract_top_level_dict(current_dict=params)

        # print('top network', param_dict.keys())
        for name, param in self.layer_dict.named_parameters():
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        out = x

        for i in range(self.num_stages):
            out = self.layer_dict['conv{}'.format(i)](out, params=param_dict['conv{}'.format(i)], training=training,
                                                      backup_running_statistics=backup_running_statistics,
                                                      num_step=num_step)
            if self.args.max_pooling:
                out = F.max_pool2d(input=out, kernel_size=(2, 2), stride=2, padding=0)

        if not self.args.max_pooling:
            out = F.avg_pool2d(out, out.shape[2])

        out = out.view(out.size(0), -1)
        out = self.layer_dict['linear'](out, param_dict['linear'])

        return out

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if (
                    param.requires_grad == True
                    and param.grad is not None
                    and torch.sum(param.grad) > 0
                ):
                    print(param.grad)
                    param.grad.zero_()
        else:
            for name, param in params.items():
                if (
                    param.requires_grad == True
                    and param.grad is not None
                    and torch.sum(param.grad) > 0
                ):
                    print(param.grad)
                    param.grad.zero_()
                    params[name].grad = None

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        for i in range(self.num_stages):
            self.layer_dict['conv{}'.format(i)].restore_backup_stats()





class sub_pixel(nn.Module):
    def __init__(self, scale, act=False):
        super(sub_pixel, self).__init__()
        modules = []
        modules.append(nn.PixelShuffle(scale))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        x = self.body(x)
        return x


class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()

        self.conv = MetaConv2dLayer(in_channels= nChannels, out_channels = growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              stride=1, use_bias=False)


    def forward(self, x, params = None):
        out = F.relu(self.conv(x, params))
        out = torch.cat((x, out), 1)
        return out


# Residual dense block (RDB) architecture
class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        self.nDenselayer = nDenselayer
        for i in range(self.nDenselayer):

            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = MetaConv2dLayer(nChannels_, nChannels, kernel_size=1, padding=0, use_bias=False)

    def forward(self, x, params = None):
        # 这里注意一下，可能出错
        out = x
        for i in range(self.nDenselayer):
            out = self.dense_layers[i](out, params['dense_layers.{}.conv.weight'.format(i)])
        out = self.conv_1x1(out, params['conv_1x1.weight'])
        out = out + x
        return out



class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv2 = MetaConv2dLayer(in_channels=2, out_channels=1, kernel_size=(1, 1), padding=0)
        self.conv2 = MetaConv2dLayer(2, 1, kernel_size=1, padding=0)
        # self.conv1 = MetaConv2dLayer(2, 2, kernel_size=(7, 7), padding=7 // 2)
        self.conv1 = MetaConv2dLayer(2, 2, kernel_size=7, padding=7 // 2)
        # self.BN = nn.BatchNorm2d(2)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x, params = None):
        if params is not None:
            params = {key: value for key, value in params.items()}
            # 9个top_layer,output_dict[top_level] = {sub_level: current_dict[key]}
            param_dict = extract_top_level_dict(current_dict=params)
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv1(result, param_dict['conv1'])
        # output = self.BN(output)
        output = self.relu(output)
        output = self.conv2(output, param_dict['conv2'])
        output = self.sigmoid(output)
        return output



class RDN(nn.Module):
    def __init__(self, args):
        super(RDN, self).__init__()
        nChannel = args.nChannel
        nDenselayer = args.nDenselayer
        nFeat = args.nFeat
        scale = args.scale
        growthRate = args.growthRate
        self.args = args
        # nn.ModuleDict 是nn.module的容器，用于包装一组网络层，以索引方式调用网络层。
        # self.layer_dict = nn.ModuleDict()

        # F-1
        self.conv1 = MetaConv2dLayer(nChannel, nFeat, kernel_size=3, padding=1, use_bias=True)
        # F0
        self.conv2 = MetaConv2dLayer(nFeat, nFeat, kernel_size=3, padding=1, use_bias=True)
        # RDBs 3
        self.RDB1 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB2 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB3 = RDB(nFeat, nDenselayer, growthRate)
        # global feature fusion (GFF)
        self.GFF_1x1 = MetaConv2dLayer(nFeat * 3, nFeat, kernel_size=1, padding=0, use_bias=True)
        self.GFF_3x3 = MetaConv2dLayer(nFeat, nFeat, kernel_size=3, padding=1, use_bias=True)
        # Upsampler
        self.conv_up = MetaConv2dLayer(nFeat, nFeat * scale * scale, kernel_size=3, padding=1, use_bias=True)
        self.upsample = sub_pixel(scale)
        # conv
        self.conv3 = MetaConv2dLayer(nFeat, 1, kernel_size=3, padding=1, use_bias=True)

        # self.sam1 = SpatialAttention()
        # self.sam2 = SpatialAttention()
        # self.sam3 = SpatialAttention()

    def forward(self, x, params):
        # 传递外部参数用，不确定能不能用
        param_dict = {}

        if params is not None:
            params = {key: value[0] for key, value in params.items()}
            # 9个top_layer,output_dict[top_level] = {sub_level: current_dict[key]}
            param_dict = extract_top_level_dict(current_dict=params)






        F_ = self.conv1(x, param_dict['conv1'])
        F_0 = self.conv2(F_, param_dict['conv2'])
        F_1 = self.RDB1(F_0, param_dict['RDB1'])
        F_2 = self.RDB2(F_1, param_dict['RDB2'])
        F_3 = self.RDB3(F_2, param_dict['RDB3'])
        FF = torch.cat((F_1, F_2, F_3), 1)
        FdLF = self.GFF_1x1(FF, param_dict['GFF_1x1'])
        FGF = self.GFF_3x3(FdLF, param_dict['GFF_3x3'])
        FDF = FGF + F_
        us = self.conv_up(FDF, param_dict['conv_up'])
        us = self.upsample(us)

        output = self.conv3(us, param_dict['conv3'])

        return output


