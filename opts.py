# -*- coding: utf-8 -*-
'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''

import argparse
import sys


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('yes', 'true', 't', '1', 'y'):
        return True
    if value in ('no', 'false', 'f', '0', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def _provided(flag):
    return any(arg == flag or arg.startswith(flag + '=') for arg in sys.argv[1:])


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', default='ravdess_preprocessing/annotations.txt', type=str, help='Annotation file path')
    parser.add_argument('--data_root', default='', type=str, help='Root directory containing the preprocessed RAVDESS ACTOR folders')
    parser.add_argument('--result_path', default='results', type=str, help='Result directory path')
    parser.add_argument('--store_name', default='model', type=str, help='Name to store checkpoints')
    parser.add_argument('--dataset', default='RAVDESS', type=str, choices=['RAVDESS', 'CMU_MOSEI', 'CREMA_D'], help='Used dataset')
    parser.add_argument('--n_classes', default=None, type=int, help='Number of classes. Inferred if not set.')
    
    parser.add_argument('--model', default='multimodalcnn', type=str, choices=['multimodalcnn', 'token_fusion_avtca'], help='Model architecture')
    parser.add_argument('--transformer_heads', default=1, type=int, help='number of transformer heads')
    parser.add_argument('--cross_attention_heads', default=1, type=int, help='number of cross-attention heads, in the paper 1 or 4')
    parser.add_argument('--num_heads', default=4, type=int, help='number of heads for token_fusion_avtca shared transformer')
    parser.add_argument('--embed_dim', default=256, type=int, help='token_fusion_avtca embedding dimension')
    parser.add_argument('--transformer_depth', default=4, type=int, help='token_fusion_avtca transformer depth')
    parser.add_argument('--fusion_tokens', default=4, type=int, help='number of learned fusion bottleneck tokens')
    parser.add_argument('--token_grid', default=4, type=int, help='visual token pooling grid size')
    parser.add_argument('--max_audio_tokens', default=128, type=int, help='maximum audio tokens')
    parser.add_argument('--max_video_tokens', default=64, type=int, help='maximum video tokens')
    parser.add_argument('--mlp_ratio', default=4, type=float, help='transformer feed-forward expansion ratio')
    parser.add_argument('--dropout', default=0.2, type=float, help='dropout for token_fusion_avtca')
    parser.add_argument('--attention_dropout', default=0.1, type=float, help='attention dropout for token_fusion_avtca')
    parser.add_argument('--audio_in_channels', default=10, type=int, help='audio tokenizer input channels before adaptation')
    parser.add_argument('--video_in_channels', default=3, type=int, help='video tokenizer input channels before adaptation')
    
    parser.add_argument('--device', default='cuda', type=str, help='Specify the device to run. Defaults to cuda, fallsback to cpu')
    
    
    parser.add_argument('--sample_size', default=224, type=int, help='Video dimensions: ravdess = 224 ')
    parser.add_argument('--sample_duration', default=15, type=int, help='Temporal duration of inputs, ravdess = 15')
    
    parser.add_argument('--learning_rate', default=0.01, type=float, help='Initial learning rate')
    parser.add_argument('--optimizer', default='adam', type=str, choices=['adam', 'adamw', 'sgd'], help='Optimizer')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--lr_steps', default=[40, 55, 65, 70, 200, 250], type=float, nargs="+", metavar='LRSteps', help='epochs to decay learning rate by 10')
    parser.add_argument('--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument('--weight_decay', default=0.001, type=float, help='Weight Decay')
    parser.add_argument('--lr_patience', default=10, type=int, help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument('--scheduler', default='auto', type=str, choices=['auto', 'step', 'plateau', 'warmup_cosine', 'none'], help='Learning-rate scheduler')
    parser.add_argument('--warmup_ratio', default=0.05, type=float, help='Warmup ratio for warmup_cosine scheduler')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch Size')
    parser.add_argument('--n_epochs', default=128, type=int, help='Number of total epochs to run')
    
    parser.add_argument('--begin_epoch', default=1, type=int, help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.')
    parser.add_argument('--resume_path', '--resume', default='', type=str, help='Checkpoint path to resume/evaluate')
    parser.add_argument('--unsafe_resume', action='store_true', help='Opt-in to use unsafe pickle-based loading for trusted resume checkpoints')
    parser.add_argument('--pretrain_path', default="EfficientFace_Trained_on_AffectNet7.pth", type=str, help='Pretrained model (.pth), efficientface')
    parser.add_argument('--no_train', action='store_true', help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument('--no_val', action='store_true', help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)
    parser.add_argument('--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test=False)
    parser.add_argument('--test_subset', default='test', type=str, help='Used subset in test (val | test)')
    
    parser.add_argument('--n_threads', default=16, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--video_norm_value', default=255, type=int, help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
 
    parser.add_argument('--manual_seed', default=None, type=int, help='Manually set random seed')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--deterministic', default=True, type=str2bool, help='Enable deterministic cuDNN behavior when possible')
    parser.add_argument('--fusion', default='it', type=str, help='fusion type: lt | it | ia')
    parser.add_argument('--mask', type=str, help='dropout type : softhard | noise | nodropout', default='softhard')
    parser.add_argument('--label_smoothing', default=0.0, type=float, help='CrossEntropy label smoothing')
    parser.add_argument('--use_class_weights', default=False, type=str2bool, help='Use inverse-frequency class weights')
    parser.add_argument('--use_focal_loss', default=False, type=str2bool, help='Use focal loss instead of standard cross entropy')
    parser.add_argument('--focal_gamma', default=2.0, type=float, help='Focal loss gamma')
    parser.add_argument('--grad_clip', default=0.0, type=float, help='Gradient clipping max norm; 0 disables clipping')
    parser.add_argument('--use_amp', default=False, type=str2bool, help='Use automatic mixed precision on CUDA')
    parser.add_argument('--debug', default=False, type=str2bool, help='Print debug tensor shapes')
    args = parser.parse_args()

    if args.manual_seed is None:
        args.manual_seed = args.seed
    else:
        args.seed = args.manual_seed

    if args.model == 'token_fusion_avtca':
        if not _provided('--optimizer'):
            args.optimizer = 'adamw'
        if not _provided('--learning_rate'):
            args.learning_rate = 1e-4
        if not _provided('--weight_decay'):
            args.weight_decay = 1e-2
        if not _provided('--label_smoothing'):
            args.label_smoothing = 0.05
        if not _provided('--use_class_weights'):
            args.use_class_weights = True
        if not _provided('--scheduler'):
            args.scheduler = 'warmup_cosine'
        if not _provided('--grad_clip'):
            args.grad_clip = 1.0
        if not _provided('--mask'):
            args.mask = 'nodropout'
    elif args.scheduler == 'auto':
        args.scheduler = 'plateau'

    return args
