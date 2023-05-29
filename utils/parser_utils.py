from torch import cuda


def get_args():
    import argparse
    import os
    import torch
    import json
    parser = argparse.ArgumentParser(description='Welcome to the MAML++ training and inference system')

    # batch_size=(self.num_of_gpus * self.batch_size * self.samples_per_iter
    parser.add_argument('--batch_size', nargs="?", type=int, default=1, help='Batch_size for experiment')
    parser.add_argument('--samples_per_iter', nargs="?", type=int, default=1)
    parser.add_argument('--max_models_to_save', nargs="?", type=int, default=3)
    parser.add_argument('--dataset_name', type=str, default="train")
    
    parser.add_argument('--experiment_name', nargs="?", type=str, default="meta_tcsr")
    parser.add_argument('--continue_from_epoch', nargs="?", type=str, default='latest',
                        help='Continue from checkpoint of epoch')
    parser.add_argument('--dropout_rate_value', type=float, default=0.2, help='Dropout_rate_value')
    parser.add_argument('--second_order', type=str, default="True", help='Dropout_rate_value')
    parser.add_argument('--first_order_to_second_order_epoch', type=int, default=1000)

    parser.add_argument('--total_epochs', type=int, default=400, help='Number of epochs per experiment')
    parser.add_argument('--total_iter_per_epoch', type=int, default=200, help='Number of iters per epoch')
    parser.add_argument('--min_learning_rate', type=float, default=0.00001, help='Min learning rate')
    parser.add_argument('--meta_learning_rate', type=float, default=0.0001, help='Learning rate of overall MAML system')
    parser.add_argument('--task_learning_rate', type=float, default=0.0001, help='Learning rate per task gradient step')
    parser.add_argument('--number_of_training_steps_per_iter', type=int, default=4,
                        help='Number of classes to sample per set')
    parser.add_argument('--number_of_evaluation_steps_per_iter', type=int, default=4,
                        help='Number of classes to sample per set')
    parser.add_argument('--num_samples_per_class', type=int, default=1,
                        help='Number of samples per set to sample')
    parser.add_argument('--num_target_samples', type=int, default=1, help='num of query')
    parser.add_argument('--multi_step_loss_num_epochs', type=int, default=15,
                        help='per_step_loss_importance_vectors')
    parser.add_argument('--use_multi_step_loss_optimization', type=bool, default=False, help='optimization')


    parser.add_argument('--image_size', type=int, default=300, help="Image size of high resolution image.")
    parser.add_argument("--gan_lr", type=float, default=0.0001,
                        help="Learning rate for gan-oral. (Default: 0.0001)")


    parser.add_argument('--nFeat', type=int, default=64, help='number of feature maps')
    parser.add_argument('--nChannel', type=int, default=3, help='number of color channels to use')
    parser.add_argument('--scale', type=int, default=2, help='scale output size /input size')
    parser.add_argument('--growthRate', type=int, default=32, help='growthRate of dense net')
    parser.add_argument('--nDenselayer', type=int, default=6, help='nDenselayer of RDB')

    parser.add_argument('--dataset_path', type=str, default="/media/aita-ocean/data/YKL/meta_train_data", help='rootpath of data')  # 自己写的需要的参数,总文件路径
    parser.add_argument('--task_path', type=str, default='/media/aita-ocean/data/YKL/meta_train_data/train', help='the path of task')
    parser.add_argument('--train_index_path', type=str, default='/media/aita-ocean/data/YKL/meta_train_data/train/CH3_TEMP_IRSPL/lr',
                        help='the path of index')
    parser.add_argument('--val_index_path', type=str, default='/media/aita-ocean/data/YKL/meta_train_data/val/CH3_TEMP_IRSPL/lr',
                        help='the path of index')
    parser.add_argument('--test_index_path', type=str, default='/media/aita-ocean/data/YKL/meta_train_data/test/CH3_TEMP_IRSPL/lr',
                        help='the path of index')

    parser.add_argument('--train_size', type=int, default=799, help='the size of train')
    parser.add_argument('--val_size', type=int, default=199, help='the path of val')
    parser.add_argument('--test_size', type=int, default=199, help='the path of test')

    parser.add_argument('--total_epochs_before_pause', type=int, default=1000, help='epochs')

    parser.add_argument('--num_evaluation_tasks', type=int, default=6, help='num of evaluation tasks')
    parser.add_argument('--test_batch_size', type=int, default=1, help='test') 



    parser.add_argument('--learnable_per_layer_per_step_inner_loop_learning_rate', type=bool, default= True,
                        help='LSLR needed')
    parser.add_argument('--enable_inner_loop_optimizable_bn_params', type=bool, default=False)
    parser.add_argument('--train_seed', type=int, default=0, help='seed')
    parser.add_argument('--val_seed', type=int, default=0, help='seed')
    parser.add_argument('--evaluate_on_test_set_only', type=bool, default=False, help='test')




    parser.add_argument('--image_height', nargs="?", type=int, default=300)
    parser.add_argument('--image_width', nargs="?", type=int, default=300)
    parser.add_argument('--image_channels', nargs="?", type=int, default=3)

    parser.add_argument('--reset_stored_filepaths', type=str, default="False")
    parser.add_argument('--reverse_channels', type=str, default="False")
    parser.add_argument('--num_of_gpus', type=int, default=1)  # gpu数目


    parser.add_argument('--labels_as_int', type=str, default="False")
    parser.add_argument('--seed', type=int, default=104)
    parser.add_argument('--gpu_to_use', type=int, default= 0) # ?
    parser.add_argument('--num_dataprovider_workers', nargs="?", type=int, default=0)
    parser.add_argument('--reset_stored_paths', type=str, default="False")
    parser.add_argument('--architecture_name', nargs="?", type=str)  # ?
    parser.add_argument('--meta_opt_bn', type=str, default="False")

    parser.add_argument('--norm_layer', type=str, default="batch_norm")  # ？
    parser.add_argument('--max_pooling', type=str, default="False")  # ？
    parser.add_argument('--per_step_bn_statistics', type=str, default="False")  #？
    parser.add_argument('--num_classes_per_set', type=int, default=20, help='Number of classes to sample per set')  # 对我的任务没什么用
    parser.add_argument('--cnn_num_blocks', type=int, default=4, help='')

    parser.add_argument('--cnn_num_filters', type=int, default=64, help='Number of classes to sample per set')
    parser.add_argument('--cnn_blocks_per_stage', type=int, default=1,
                        help='Number of classes to sample per set')
    parser.add_argument('--name_of_args_json_file', type=str, default="None")






    args = parser.parse_args()
    args_dict = vars(args)
    if args.name_of_args_json_file != "None":
        args_dict = extract_args_from_json(args.name_of_args_json_file, args_dict)

    for key in list(args_dict.keys()):

        if str(args_dict[key]).lower() == "true":
            args_dict[key] = True
        elif str(args_dict[key]).lower() == "false":
            args_dict[key] = False


        print(key, args_dict[key], type(args_dict[key]))

    args = Bunch(args_dict)


    args.use_cuda = torch.cuda.is_available()
    if torch.cuda.is_available():
        device = torch.cuda.current_device()

        print("use GPU", device)
        print("GPU ID {}".format(torch.cuda.current_device()))

    else:
        print("use CPU")
        device = torch.device('cpu')


    return args, device



class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)

def extract_args_from_json(json_file_path, args_dict):
    import json
    summary_filename = json_file_path
    with open(summary_filename) as f:
        summary_dict = json.load(fp=f)

    for key in summary_dict.keys():
        if "continue_from" not in key and "gpu_to_use" not in key:
            args_dict[key] = summary_dict[key]

    return args_dict





