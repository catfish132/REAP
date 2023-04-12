import argparse
from utils.config import CfgNode


def parse_agrs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss_weight', type=str, default='abc')
    parser.add_argument('--a', type=float, default=1.0)
    parser.add_argument('--b', type=float, default=1.0)
    parser.add_argument('--c', type=float, default=1.0)
    parser.add_argument('--model_type', type=str, default='autoencoder',
                        help='选择使用的模型。可用：autoencoder,transformer,selfmatcher')
    parser.add_argument('--description', type=str, default='no description')
    parser.add_argument('--num_slot', type=int, default=18, help='slot 的数量')
    # Data input settings
    parser.add_argument('--image_dir', type=str, default='/home/jinyuda/dataset/medical/iu_xray/images/',
                        help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='/home/jinyuda/dataset/medical/iu_xray/new_annotation.json',
                        help='the path to the directory containing the data.')
    parser.add_argument('--visual_path', type=str, default="/ssd2/jyd_data/dataset/iu_xray/iu_xray_visual.npy")
    parser.add_argument('--tag_path', type=str, default="/ssd2/jyd_data/dataset/iu_xray/IU_tag_id.npy")
    parser.add_argument('--tag_size', type=int, default=48, help='48 for iu_xray with threshold=30')
    parser.add_argument('--tag_topk', type=int, default=5, help='48 for iu_xray with threshold=30')
    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'],
                        help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    # Model settings (for Transformer)
    parser.add_argument('--label_smoothing', type=float, default=0,
                        help='')
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=2048, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search',
                        help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/iu_xray', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/',
                        help='the patch to save the results of experiments')
    parser.add_argument('--log_period', type=int, default=10, help='the logging interval (in batches).')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'],
                        help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    # Optimization: for the Language Model from luoruotian
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=1e-4, help='the learning rate for the remaining parameters.')
    # optimizer parameters for original training
    parser.add_argument('--optim', type=str, default='adam',
                        help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam|adamw')
    parser.add_argument('--optim_alpha', type=float, default=0.9,
                        help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999,
                        help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                        help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--weight_decay', type=float, default=5e-5,
                        help='weight_decay')
    # optimizer for rl training
    parser.add_argument('--adam_betas', type=tuple, default=(0.9, 0.98), help='the weight decay.')
    parser.add_argument('--adam_eps', type=float, default=1e-9, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')
    parser.add_argument('--noamopt_warmup', type=int, default=5000, help='.')
    parser.add_argument('--noamopt_factor', type=int, default=1, help='.')
    # for RL
    parser.add_argument('--train_sample_n', type=int, default=1, help='The reward weight from cider')
    parser.add_argument('--train_sample_method', type=str, default='sample', help='')
    parser.add_argument('--train_beam_size', type=int, default=1, help='')
    parser.add_argument('--sc_sample_method', type=str, default='greedy', help='')
    parser.add_argument('--sc_beam_size', type=int, default=1, help='')
    parser.add_argument('--sc_eval_period', type=int, default=10000, help='the saving period (in epochs).')
    parser.add_argument('--reduce_on_plateau_factor', type=float, default=0.5, help='')
    parser.add_argument('--reduce_on_plateau_patience', type=int, default=3, help='')
    # lr scheduler only for original training
    parser.add_argument('--lr_scheduler', type=str, default='StepLR',
                        help='choice from |reduce_on_plateau||StepLR||noamopt|')
    # StepLR
    parser.add_argument('--StepLR_step', type=int, default=1,
                        help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--StepLR_rate', type=float, default=0.8,
                        help='every how many iterations thereafter to drop LR?(in epoch)')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--resume', type=str, help='checkpoint path to continue training ')
    parser.add_argument('--load', type=str, help='path to load checkpoints while testing')
    # config
    parser.add_argument('--cfg', type=str, default=None,
                        help='configuration; similar to what is used in detectron')
    # How will config be used
    # 1) read cfg argument, and load the cfg file if it's not None
    # 2) Overwrite cfg argument with set_cfgs
    # 3) parse config argument to args.
    # 4) in the end, parse command line argument and overwrite args

    args = parser.parse_args()
    if args.cfg is not None:

        cn = CfgNode(CfgNode.load_yaml_with_base(args.cfg))
        for k, v in cn.items():
            if not hasattr(args, k):
                print('Warning: key %s not in args' % k)
            setattr(args, k, v)
        args = parser.parse_args(namespace=args)

    return args
