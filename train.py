# -*- coding: utf-8 -*
# @Time : 2022/3/15 12:02
# @Author : 杨坤林
# @File : train.py
# @Software : PyCharm

from data import MetaLearningSystemDataLoader
from experiment_builder import ExperimentBuilder
from maml_tcsr import MAML_TCSR
from utils.parser_utils import get_args
from utils.dataset_tools import maybe_unzip_dataset


args, device = get_args()
model = MAML_TCSR(args=args, device=device,
                              im_shape=(2, args.image_channels,
                                        args.image_height         , args.image_width))

data = MetaLearningSystemDataLoader

maml_system = ExperimentBuilder(model=model, data=data, args=args, device=device)
maml_system.run_experiment()