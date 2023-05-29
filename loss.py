# -*- coding: utf-8 -*
# @Time : 2022/7/14 17:52
# @Author : 杨坤林
# @File : loss.py
# @Software : PyCharm
import lpips
import torch
import torch.nn.functional
import torchvision

__all__ = [
    "LPIPSLoss", "TVLoss", "VGGLoss"
]


class LPIPSLoss(torch.nn.Module):


    def __init__(self) -> None:
        super(LPIPSLoss, self).__init__()
        self.model = lpips.LPIPS(net="vgg", verbose=False).eval()

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        lpips_loss = torch.mean(self.model(source, target))

        return lpips_loss



class TVLoss(torch.nn.Module):


    def __init__(self, weight: torch.Tensor) -> None:

        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        tv_loss = self.weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

        return tv_loss

    @staticmethod
    def tensor_size(t):
        return t.image_size()[1] * t.image_size()[2] * t.image_size()[3]

class VGGLoss(torch.nn.Module):


    def __init__(self, feature_layer: int = 35) -> None:

        super(VGGLoss, self).__init__()
        model = torchvision.models.vgg19(pretrained=True)
        self.features = torch.nn.Sequential(*list(model.features.children())[:feature_layer]).eval()
        # Freeze parameters. Don't train.
        for name, param in self.features.named_parameters():
            param.requires_grad = False

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        vgg_loss = torch.nn.functional.l1_loss(self.features(source), self.features(target))

        return vgg_loss