# -*- coding: utf-8 -*-
'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''

import argparse


def parse_opts(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', default='preprocessing/ravdess/annotations.txt', type=str, help='Annotation file path')
    parser.add_argument('--data_root', default='', type=str, help='Root directory containing the preprocessed RAVDESS ACTOR folders')
    parser.add_argument('--result_path', default='results', type=str, help='Result directory path')
    parser.add_argument('--store_name', default='model', type=str, help='Name to store checkpoints')
    parser.add_argument('--dataset', default='RAVDESS', type=str, choices=['RAVDESS', 'CREMAD', 'ENGAGENET', 'DAISEE'], help='Dataset name.')
    parser.add_argument('--n_classes', default=8, type=int, help='Number of output classes. Use 4 for the bootstrap ENGAGENET engagement setup.')
    
    parser.add_argument('--model', default='multimodal_cnn', type=str, choices=['multimodal_cnn'], help='Model architecture.')
    parser.add_argument('--audio_features', default='mel', type=str, choices=['mfcc', 'mel'], help='Audio feature type. mel=64-channel mel spectrogram (best), mfcc=10-channel MFCC')
    parser.add_argument('--spec_augment', action='store_true', help='Apply SpecAugment-style time/frequency masking to training audio features')
    parser.set_defaults(spec_augment=False)
    parser.add_argument('--spec_time_masks', default=2, type=int, help='Number of SpecAugment time masks per training sample')
    parser.add_argument('--spec_freq_masks', default=2, type=int, help='Number of SpecAugment frequency masks per training sample')
    parser.add_argument('--spec_time_mask_width', default=20, type=int, help='Maximum SpecAugment time-mask width in feature frames')
    parser.add_argument('--spec_freq_mask_width', default=8, type=int, help='Maximum SpecAugment frequency-mask width in mel/MFCC bins')
    parser.add_argument('--audio_channel_attention', action='store_true', help='Apply a lightweight channel-attention gate after the audio stage-2 encoder')
    parser.set_defaults(audio_channel_attention=False)
    parser.add_argument('--max_text_tokens', default=32, type=int, help='Maximum hashed text tokens kept per sample when annotations include optional text.')
    parser.add_argument('--text_vocab_size', default=4096, type=int, help='Hash-bucket vocabulary size for optional text add-on tokens.')
    parser.add_argument('--late_text_fusion', action='store_true', help='Enable the late optional text-fusion add-on after audio/video pooling.')
    parser.add_argument('--no_late_text_fusion', dest='late_text_fusion', action='store_false', help='Disable late text fusion for AV-only checkpoints.')
    parser.set_defaults(late_text_fusion=True)
    parser.add_argument('--behavior', action='store_true', help='Enable the numeric-AU behavior modality (it fusion only).')
    parser.set_defaults(behavior=False)
    parser.add_argument('--behavior_feature_dim', default=22, type=int, help='Numeric behavior feature dim (17 AU + gaze2 + pose3).')
    parser.add_argument('--behavior_skip_dim', default=64, type=int, help='Width of the direct pooled-AU skip into the classifier.')
    parser.add_argument('--behavior_dir', default='', type=str, help='Directory of per-clip behavior .npy files (OpenFace features).')
    parser.add_argument('--behavior_baselines', default='', type=str, help='JSON of per-subject neutral AU baselines.')
    parser.add_argument('--text_fusion', action='store_true', help='Fuse behavior-caption text via the sentence TextEncoder (it fusion only). Replaces the hashed late-text add-on.')
    parser.set_defaults(text_fusion=False)
    parser.add_argument('--text_backend', default='hashing', choices=['hashing', 'minilm', 'auto'], help='Sentence embedding backend for text fusion. hashing = dependency-free (default).')
    parser.add_argument('--num_heads', default=1, type=int, help='number of heads, in the paper 1 or 4')
    
    parser.add_argument('--device', default='cuda', type=str, help='Specify the device to run. Defaults to cuda, fallsback to cpu')
    
    
    parser.add_argument('--sample_size', default=224, type=int, help='Video dimensions: ravdess = 224 ')
    parser.add_argument('--sample_duration', default=15, type=int, help='Deprecated fixed clip length alias. Use --max_video_frames for dynamic-length runs.')
    parser.add_argument('--max_video_frames', default=96, type=int, help='Maximum visual frames kept per sample after temporal sampling/padding.')
    parser.add_argument('--max_audio_steps', default=0, type=int, help='Maximum audio feature time steps kept per sample. 0 keeps all available steps.')
    parser.add_argument('--frame_sampling', default='uniform', choices=['uniform', 'stride'], help='Deterministic temporal subsampling policy used for validation/test and as the default training policy.')
    parser.add_argument('--train_frame_sampling', default='', choices=['', 'uniform', 'stride', 'random'], help='Optional training-only temporal sampling policy. Use random for crop-style augmentation on long clips.')
    parser.add_argument('--temporal_pad_value', default=0.0, type=float, help='Pad value used for temporal audio/video batching.')
    parser.add_argument('--full_video_preprocessing', action='store_true', help='Use full-length EngageNet/DAiSEE preprocessing instead of the legacy 3.6s/15-frame contract.')
    parser.set_defaults(full_video_preprocessing=False)
    
    parser.add_argument('--learning_rate', default=0.06, type=float, help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'adamw'], help='Optimizer used for training.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--lr_steps', default=[40, 55, 65, 70, 200, 250], type=int, nargs="+", metavar='LRSteps', help='epochs to decay learning rate by 10')
    parser.add_argument('--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight Decay')
    parser.add_argument('--lr_scheduler', default='step', choices=['step', 'plateau', 'warmup_cosine'], help='Learning-rate schedule. step uses --lr_steps; plateau uses validation loss; warmup_cosine steps every training batch.')
    parser.add_argument('--lr_patience', default=10, type=int, help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument('--warmup_ratio', default=0.05, type=float, help='Fraction of warmup steps for --lr_scheduler warmup_cosine.')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch Size')
    parser.add_argument('--n_epochs', default=10, type=int, help='Number of total epochs to run')
    parser.add_argument('--early_stopping_patience', default=0, type=int, help='Stop after this many validation epochs without selection-metric improvement. 0 disables early stopping.')
    parser.add_argument('--grad_clip_norm', default=0.0, type=float, help='Optional max gradient norm. 0 disables clipping.')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='Accumulate gradients across this many batches before each optimizer step.')
    parser.add_argument('--ema_decay', default=0.0, type=float, help='Optional exponential moving average decay for model weights. 0 disables EMA.')
    parser.add_argument('--label_smoothing', default=0.1, type=float, help='Cross-entropy label smoothing. 0 disables smoothing.')
    parser.add_argument('--loss', default='ce', choices=['ce', 'focal', 'ordinal_distance'], help='Classification loss. focal emphasizes hard examples; ordinal_distance adds an ordered-class distance penalty for engagement levels.')
    parser.add_argument('--focal_gamma', default=2.0, type=float, help='Focusing parameter for --loss focal. Higher values down-weight easy examples more strongly.')
    parser.add_argument('--ordinal_distance_weight', default=0.35, type=float, help='Weight for the ordinal expected-distance penalty when --loss ordinal_distance is used.')
    parser.add_argument('--class_weighting', default='none', choices=['none', 'inverse', 'sqrt_inverse'], help='Optional class-weighted loss from training label counts.')
    parser.add_argument('--class_balance_sampler', default='none', choices=['none', 'inverse', 'sqrt_inverse'], help='Optional weighted sampler for imbalanced training classes.')
    parser.add_argument('--prediction_mode', default='argmax', choices=['argmax', 'expected_round'], help='Prediction decoding for ordered labels. expected_round rounds the softmax expected class index.')
    parser.add_argument('--selection_min_delta', default=0.0, type=float, help='Minimum validation selection-metric improvement required to update best checkpoint or reset early stopping.')
    parser.add_argument(
        '--selection_metric',
        default='top1_accuracy',
        choices=['top1_accuracy', 'balanced_accuracy', 'uar', 'adjacent_accuracy', 'mean_absolute_class_error', 'loss', 'f1_macro', 'f1_weighted'],
        help='Validation metric used to choose the best checkpoint. Lower is better for loss and mean_absolute_class_error.',
    )
    
    parser.add_argument('--begin_epoch', default=1, type=int, help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.')
    parser.add_argument('--resume_path', default='', type=str, help='Save data (.pth) of previous training')
    parser.add_argument('--pretrain_path', default="pretrained/EfficientFace_Trained_on_AffectNet7.pth", type=str, help='Pretrained model (.pth), efficientface')
    parser.add_argument(
        '--visual_backbone',
        default='efficientface',
        choices=['efficientface', 'attention_local'],
        help='Visual feature extractor: efficientface keeps the existing path; attention_local uses the explicit channel/spatial/local branch',
    )
    parser.add_argument(
        '--visual_stem_pooling',
        default='maxpool',
        choices=['maxpool', 'stride_conv'],
        help='Downsampling method after the first visual conv in the attention_local backbone',
    )
    parser.add_argument('--checkpoint_path', default='', type=str, help='Explicit checkpoint to evaluate during test-only flows.')
    parser.add_argument('--no_train', action='store_true', help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument('--no_val', action='store_true', help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)
    parser.add_argument('--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test=False)
    parser.add_argument('--test_subset', default='test', type=str, choices=['test', 'val'], help='Used subset in test (val | test)')
    
    parser.add_argument('--n_threads', default=16, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--video_norm_value', default=255, type=int, help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument('--max_train_batches', default=0, type=int, help='Optional cap on training batches per epoch for smoke tests. 0 means all batches.')
    parser.add_argument('--max_val_batches', default=0, type=int, help='Optional cap on validation batches for smoke tests. 0 means all batches.')
 
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--fusion', default='it', type=str, choices=['lt', 'it', 'ia'], help='fusion type: lt | it | ia')
    parser.add_argument('--mask', type=str, choices=['softhard', 'noise', 'nodropout'], help='dropout type : softhard | noise | nodropout', default='softhard')
    args = parser.parse_args(argv)

    return args
