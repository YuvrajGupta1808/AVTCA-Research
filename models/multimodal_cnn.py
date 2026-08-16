import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modulator import Channel, Spatial, Modulator
from models.efficient_face import LocalFeatureExtractor, InvertedResidual
from models.transformer import AttentionBlock, Attention
from models.behavior_encoder import BehaviorEncoder
from models.text_encoder import TextEncoder
from torch.nn import MultiheadAttention


class AttentionPool(nn.Module):
    """Learned weighted sum over a temporal sequence — preserves which time steps matter."""
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, 1)

    def forward(self, x, mask=None):
        # x: (B, T, dim)
        logits = self.proj(x)
        if mask is not None:
            invalid = (~mask).unsqueeze(-1)
            logits = logits.masked_fill(invalid, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1)   # (B, T, 1)
        if mask is not None:
            weights = weights * mask.unsqueeze(-1).to(weights.dtype)
            denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            weights = weights / denom
        return (weights * x).sum(dim=1)                # (B, dim)


class TemporalChannelGate(nn.Module):
    """Squeeze-and-excitation gate for temporal feature maps."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)

def conv1d_block(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,stride=stride, padding=padding),nn.BatchNorm1d(out_channels),
                                   nn.ReLU(inplace=True)) 

class EfficientFaceTemporal(nn.Module):

    def __init__(self, stages_repeats, stages_out_channels, num_classes=7, im_per_sample=25):
        super(EfficientFaceTemporal, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(output_channels),
                                   nn.ReLU(inplace=True),)
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        self.local = LocalFeatureExtractor(29, 116, 1)
        self.modulator = Modulator(29)

        output_channels = self._stage_out_channels[-1]

        self.conv5 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
                                   nn.BatchNorm2d(output_channels),
                                   nn.ReLU(inplace=True),)
        self.conv1d_0 = conv1d_block(output_channels, 64)
        self.conv1d_1 = conv1d_block(64, 64)
        self.conv1d_2 = conv1d_block(64, 128)
        self.conv1d_3 = conv1d_block(128, 128)

        self.classifier_1 = nn.Sequential(
                nn.Linear(128, num_classes),
            )
        self.im_per_sample = im_per_sample
        
    def forward_features(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        # Modulator (channel + spatial attention) and LocalFeatureExtractor both run on
        # the raw post-maxpool features. Their combined output feeds into the Inverted
        # Residual blocks, matching the diagram order: attend first, then process deeply.
        x = self.stage2(self.modulator(x)) + self.local(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3]) #global average pooling
        return x

    def forward_stage1(self, x):
        #Getting samples per batch
        if x.shape[0] % self.im_per_sample != 0:
            raise ValueError(
                f'Visual feature batch has {x.shape[0]} frames, which is not divisible by '
                f'im_per_sample={self.im_per_sample}.'
            )
        n_samples = x.shape[0] // self.im_per_sample
        x = x.view(n_samples, self.im_per_sample, x.shape[1])
        return self.forward_stage1_from_sequence(x)

    def forward_stage1_from_sequence(self, x):
        x = x.permute(0,2,1)
        x = self.conv1d_0(x)
        x = self.conv1d_1(x)
        return x
        
        
    def forward_stage2(self, x):
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)
        return x
    
    def forward_classifier(self, x):
        x = x.mean([-1]) #pooling accross temporal dimension
        x1 = self.classifier_1(x)
        return x1
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_stage1(x)
        x = self.forward_stage2(x)
        x = self.forward_classifier(x)
        return x
        
      

class AttentionLocalVisualTemporal(nn.Module):
    """Diagram-style visual branch with explicit channel, spatial, and local paths.

    This keeps the same public interface as EfficientFaceTemporal so the rest of
    the audiovisual fusion stack can switch visual extractors without changes.
    """

    def __init__(
        self,
        stages_repeats,
        stages_out_channels,
        num_classes=7,
        im_per_sample=25,
        stem_pooling='maxpool',
    ):
        super(AttentionLocalVisualTemporal, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        if stem_pooling not in ['maxpool', 'stride_conv']:
            raise ValueError('expected stem_pooling to be maxpool or stride_conv')

        input_channels = 3
        stem_channels = stages_out_channels[0]
        local_channels = stages_out_channels[1]
        self.stem_pooling = stem_pooling

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, stem_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )
        if stem_pooling == 'maxpool':
            self.stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.stem_pool = nn.Sequential(
                nn.Conv2d(stem_channels, stem_channels, 3, 2, 1, bias=False),
                nn.BatchNorm2d(stem_channels),
                nn.ReLU(inplace=True),
            )

        self.channel_att = Channel(stem_channels)
        self.spatial_att = Spatial(stem_channels)
        self.local = LocalFeatureExtractor(stem_channels, local_channels, 1)

        # Smaller than EfficientFace stage2, but still produces the same feature
        # shape as the local branch so both paths can be fused by addition.
        attention_blocks = [InvertedResidual(stem_channels, local_channels, 2)]
        for _ in range(max(stages_repeats[0] - 1, 0)):
            attention_blocks.append(InvertedResidual(local_channels, local_channels, 1))
        self.attention_reduce = nn.Sequential(*attention_blocks)

        input_channels = local_channels
        for name, repeats, output_channels in zip(
            ['stage3', 'stage4'],
            stages_repeats[1:],
            stages_out_channels[2:4],
        ):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for _ in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = stages_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.conv1d_0 = conv1d_block(output_channels, 64)
        self.conv1d_1 = conv1d_block(64, 64)
        self.conv1d_2 = conv1d_block(64, 128)
        self.conv1d_3 = conv1d_block(128, 128)
        self.classifier_1 = nn.Sequential(nn.Linear(128, num_classes))
        self.im_per_sample = im_per_sample

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.stem_pool(x)

        channel_map = self.channel_att(x)
        spatial_map = self.spatial_att(x)
        attended = torch.sigmoid(channel_map * spatial_map) * x

        x = self.attention_reduce(attended) + self.local(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])
        return x

    def forward_stage1(self, x):
        if x.shape[0] % self.im_per_sample != 0:
            raise ValueError(
                f'Visual feature batch has {x.shape[0]} frames, which is not divisible by '
                f'im_per_sample={self.im_per_sample}.'
            )
        n_samples = x.shape[0] // self.im_per_sample
        x = x.view(n_samples, self.im_per_sample, x.shape[1])
        return self.forward_stage1_from_sequence(x)

    def forward_stage1_from_sequence(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1d_0(x)
        x = self.conv1d_1(x)
        return x

    def forward_stage2(self, x):
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)
        return x

    def forward_classifier(self, x):
        x = x.mean([-1])
        x1 = self.classifier_1(x)
        return x1

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_stage1(x)
        x = self.forward_stage2(x)
        x = self.forward_classifier(x)
        return x


def init_feature_extractor(model, path):
    if path == 'None' or path is None:
        return
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    pre_trained_dict = checkpoint['state_dict']
    pre_trained_dict = {key.replace("module.", ""): value for key, value in pre_trained_dict.items()}
    # Load only weights whose shapes match — newer PyTorch raises on shape mismatch even with strict=False
    model_dict = model.state_dict()
    compatible = {k: v for k, v in pre_trained_dict.items()
                  if k in model_dict and model_dict[k].shape == v.shape}
    skipped = len([k for k in pre_trained_dict if k in model_dict and model_dict[k].shape != pre_trained_dict[k].shape])
    model_dict.update(compatible)
    print(f'Initializing efficientnet... loaded {len(compatible)} layers, skipped {skipped} shape-mismatched')
    model.load_state_dict(model_dict)
    print('Loaded efficientnet checkpoint...')

    
def conv2d_block_audio(in_channels, out_channels, kernel_size=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
    )

def conv1d_block_audio(in_channels, out_channels, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding='same'),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(2, 2),
    )

class AudioCNNPool(nn.Module):

    def __init__(self, num_classes=8):
        super(AudioCNNPool, self).__init__()

        self.conv2d_0 = conv2d_block_audio(1, 64)
        self.conv2d_1 = conv2d_block_audio(64, 128)
        self.conv1d_2 = conv1d_block_audio(128, 256)
        self.conv1d_3 = conv1d_block_audio(256, 128)
        
        self.classifier_1 = nn.Sequential(
                nn.Linear(128, num_classes),
            )
            
    def forward(self, x):
        x = self.forward_stage1(x)
        x = self.forward_stage2(x)
        x = self.forward_classifier(x)
        return x


    def forward_stage1(self, x):
        x = x.unsqueeze(1)      # (B, 1, 10, T)
        x = self.conv2d_0(x)    # (B, 64, 5, T//2)
        x = self.conv2d_1(x)    # (B, 128, ~2, T//4)
        x = x.mean(dim=2)       # collapse freq → (B, 128, T//4)
        return x
    
    def forward_stage2(self,x):
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)   
        return x
    
    def forward_classifier(self, x):   
        x = x.mean([-1]) #pooling accross temporal dimension
        x1 = self.classifier_1(x)
        return x1

    


class LateTextFusion(nn.Module):
    """Refine an audio-visual summary by reading text afterwards."""

    def __init__(self, embed_dim, vocab_size, num_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.GRU(
            input_size=embed_dim,
            hidden_size=embed_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.readout = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, av_context, text_tokens=None, text_mask=None):
        if text_tokens is None or text_mask is None:
            return av_context

        valid_rows = text_mask.any(dim=1)
        if not valid_rows.any():
            return av_context

        refined_context = av_context.clone()
        valid_tokens = text_tokens[valid_rows]
        valid_mask = text_mask[valid_rows]
        valid_context = av_context[valid_rows]

        text_embeddings = self.embedding(valid_tokens)
        encoded_text, _ = self.encoder(text_embeddings)
        attended_text, _ = self.readout(
            query=valid_context.unsqueeze(1),
            key=encoded_text,
            value=encoded_text,
            key_padding_mask=~valid_mask,
        )
        text_context = attended_text.squeeze(1)
        gate = self.gate(torch.cat((valid_context, text_context), dim=-1))
        refined_context[valid_rows] = self.norm(valid_context + gate * text_context)
        return refined_context


class MultiModalCNN(nn.Module):
    def __init__(
        self,
        num_classes=8,
        fusion='it',
        seq_length=15,
        pretr_ef='None',
        num_heads=1,
        audio_channel_attention=False,
        visual_backbone='efficientface',
        visual_stem_pooling='maxpool',
        it_fusion_mode='modern',
        text_vocab_size=4096,
        late_text_fusion=True,
        behavior=False,
        behavior_feature_dim=22,
        behavior_skip_dim=64,
        text_fusion=False,
        text_backend='hashing',
    ):
        super(MultiModalCNN, self).__init__()
        if fusion not in ['ia', 'it', 'lt']:
            raise ValueError(f'Unsupported fusion method "{fusion}". Expected one of: ia, it, lt')
        if it_fusion_mode not in ['modern', 'legacy']:
            raise ValueError(f'Unsupported it_fusion_mode "{it_fusion_mode}". Expected one of: legacy, modern')

        self.audio_model = AudioCNNPool(num_classes=num_classes)
        self.visual_backbone = visual_backbone
        self.it_fusion_mode = it_fusion_mode
        self.late_text_fusion = late_text_fusion
        if visual_backbone == 'efficientface':
            self.visual_model = EfficientFaceTemporal([4, 8, 4], [29, 116, 232, 464, 1024], num_classes, seq_length)
        elif visual_backbone == 'attention_local':
            self.visual_model = AttentionLocalVisualTemporal(
                [2, 4, 2],
                [29, 116, 232, 464, 1024],
                num_classes,
                seq_length,
                stem_pooling=visual_stem_pooling,
            )
        else:
            raise ValueError(
                'Unsupported visual_backbone "{}". Expected efficientface or attention_local.'.format(
                    visual_backbone
                )
            )

        init_feature_extractor(self.visual_model, pretr_ef)
                           
        e_dim = 128
        input_dim_video = 128
        input_dim_audio = 128
        self.fusion=fusion
        self.audio_channel_attention = audio_channel_attention
        self.audio_feature_gate = TemporalChannelGate(e_dim) if audio_channel_attention else nn.Identity()

        if fusion in ['lt', 'it']:
            if fusion  == 'lt':
                self.av = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=e_dim, num_heads=num_heads)
                self.va = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=e_dim, num_heads=num_heads)
            elif fusion == 'it':
                input_dim_video = input_dim_video // 2
                # Subsample audio to match video's temporal length before cross-attention
                self.audio_temporal_pool = nn.AdaptiveAvgPool1d(seq_length)
                self.av1 = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio, num_heads=num_heads)
                self.va1 = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video, num_heads=num_heads)
                self.audioCrossAttention  = AttentionBlock(in_dim_k=e_dim, in_dim_q=e_dim, out_dim=e_dim, num_heads=num_heads)
                self.visualCrossAttention = AttentionBlock(in_dim_k=e_dim, in_dim_q=e_dim, out_dim=e_dim, num_heads=num_heads)
                if it_fusion_mode == 'modern':
                    self.attn_dropout = nn.Dropout(0.1)
                    self.attn_pool_audio = AttentionPool(e_dim)
                    self.attn_pool_video = AttentionPool(e_dim)
        
        elif fusion in ['ia']:
            input_dim_video = input_dim_video // 2
            
            self.av1 = Attention(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio, num_heads=num_heads)
            self.va1 = Attention(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video, num_heads=num_heads)

            
        self.audioAttention  = MultiheadAttention(e_dim, num_heads)
        self.visualAttention = MultiheadAttention(e_dim, num_heads)
        if late_text_fusion:
            self.av_context = nn.Sequential(
                nn.Linear(e_dim * 2, e_dim),
                nn.ReLU(inplace=True),
            )
            self.text_addon = LateTextFusion(e_dim, text_vocab_size, num_heads)

        self.classifier_1 = nn.Sequential(
            nn.Linear(e_dim*2, num_classes),
        )

        self.behavior = behavior
        self.text_fusion = text_fusion
        self.behavior_skip_dim = behavior_skip_dim
        fused_extra = 0
        if (behavior or text_fusion) and it_fusion_mode == 'legacy':
            raise ValueError("behavior/text fusion require it_fusion_mode='modern' (the legacy path skips fusion)")
        if behavior:
            if fusion != 'it':
                raise ValueError("behavior fusion is only supported with fusion='it'")
            self.behavior_encoder = BehaviorEncoder(feature_dim=behavior_feature_dim, hidden=e_dim)
            self.behavior_av_proj = nn.Linear(e_dim * 2, e_dim)
            self.behaviorCrossAttention = MultiheadAttention(e_dim, num_heads, batch_first=True)
            self.behavior_skip = nn.Sequential(
                nn.Linear(behavior_feature_dim, behavior_skip_dim),
                nn.ReLU(inplace=True),
            )
            self.behavior_missing = nn.Parameter(torch.randn(e_dim))
            fused_extra += e_dim + behavior_skip_dim
        if text_fusion:
            if fusion != 'it':
                raise ValueError("text fusion is only supported with fusion='it'")
            self.text_encoder = TextEncoder(embed_dim=e_dim, backend=text_backend)
            self.text_av_proj = nn.Linear(e_dim * 2, e_dim)
            self.textCrossAttention = MultiheadAttention(e_dim, num_heads, batch_first=True)
            self.text_missing = nn.Parameter(torch.randn(e_dim))
            fused_extra += e_dim
        if fused_extra:
            self.classifier_fused = nn.Linear(e_dim * 2 + fused_extra, num_classes)

    def _ensure_video_mask(self, x_visual, video_mask):
        if video_mask is not None:
            return video_mask.bool()
        return torch.ones(x_visual.shape[0], x_visual.shape[1], dtype=torch.bool, device=x_visual.device)

    def _visual_backbone_features(self, x_visual):
        if x_visual.dim() == 5:
            batch_size, time_steps, channels, height, width = x_visual.shape
            flattened = x_visual.reshape(batch_size * time_steps, channels, height, width)
            features = self.visual_model.forward_features(flattened)
            return features.view(batch_size, time_steps, -1)
        if x_visual.dim() == 4:
            features = self.visual_model.forward_features(x_visual)
            return features
        raise ValueError(f'Unsupported visual input rank: {x_visual.dim()}')

    def _visual_stage1(self, visual_features):
        if visual_features.dim() == 3:
            return self.visual_model.forward_stage1_from_sequence(visual_features)
        return self.visual_model.forward_stage1(visual_features)

    def _mask_lengths(self, mask):
        return mask.long().sum(dim=1)

    def _adaptive_align_audio_to_video(self, x_audio, audio_lengths, video_lengths, target_length):
        aligned = []
        aligned_mask = []
        max_audio_length = x_audio.shape[-1]
        for sample, audio_length, video_length in zip(x_audio, audio_lengths.tolist(), video_lengths.tolist()):
            audio_length = min(max(int(audio_length), 1), max_audio_length)
            video_length = min(max(int(video_length), 1), target_length)
            sample_valid = sample[:, :audio_length].unsqueeze(0)
            pooled = torch.nn.functional.adaptive_avg_pool1d(sample_valid, video_length).squeeze(0)
            if video_length < target_length:
                pad = sample.new_zeros(sample.shape[0], target_length - video_length)
                pooled = torch.cat([pooled, pad], dim=1)
            aligned.append(pooled)
            mask = torch.zeros(target_length, dtype=torch.bool, device=sample.device)
            mask[:video_length] = True
            aligned_mask.append(mask)
        return torch.stack(aligned, dim=0), torch.stack(aligned_mask, dim=0)

    def _downsample_mask(self, mask, levels=1):
        pooled = mask.to(dtype=torch.float32).unsqueeze(1)
        for _ in range(levels):
            pooled = F.max_pool1d(pooled, kernel_size=2, stride=2)
        return pooled.squeeze(1) > 0

    def forward(
        self,
        x_audio,
        x_visual,
        audio_mask=None,
        video_mask=None,
        audio_lengths=None,
        video_lengths=None,
        text_tokens=None,
        text_mask=None,
        behavior_feats=None,
        behavior_present=None,
    ):

        if self.fusion == 'lt':
            return self.forward_transformer(x_audio, x_visual)

        elif self.fusion == 'ia':
            return self.forward_feature_2(x_audio, x_visual)
       
        elif self.fusion == 'it':
            return self.forward_feature_3(
                x_audio,
                x_visual,
                audio_mask=audio_mask,
                video_mask=video_mask,
                audio_lengths=audio_lengths,
                video_lengths=video_lengths,
                text_tokens=text_tokens,
                text_mask=text_mask,
                behavior_feats=behavior_feats,
                behavior_present=behavior_present,
            )



    def forward_feature_3(
        self,
        x_audio,
        x_visual,
        audio_mask=None,
        video_mask=None,
        audio_lengths=None,
        video_lengths=None,
        text_tokens=None,
        text_mask=None,
        behavior_feats=None,
        behavior_present=None,
    ):
        if self.it_fusion_mode == 'legacy':
            return self.forward_feature_3_legacy(x_audio, x_visual)

        x_audio = self.audio_model.forward_stage1(x_audio)
        if x_visual.dim() == 5:
            x_visual = self._visual_backbone_features(x_visual)
            x_visual = self._visual_stage1(x_visual)
        else:
            x_visual = self.visual_model.forward_features(x_visual)
            x_visual = self.visual_model.forward_stage1(x_visual)

        video_mask = self._ensure_video_mask(x_visual if x_visual.dim() == 3 else x_visual.permute(0, 2, 1), video_mask)
        if video_lengths is None:
            video_lengths = self._mask_lengths(video_mask)
        if audio_mask is not None:
            audio_stage1_mask = self._downsample_mask(audio_mask.to(x_audio.device), levels=2)
            audio_stage1_lengths = self._mask_lengths(audio_stage1_mask)
        elif audio_lengths is not None:
            audio_stage1_lengths = torch.div(audio_lengths.to(x_audio.device), 4, rounding_mode='floor').clamp_min(1)
        else:
            audio_stage1_lengths = torch.full(
                (x_audio.shape[0],),
                x_audio.shape[-1],
                dtype=torch.long,
                device=x_audio.device,
            )

        # Align audio temporal dimension to each sample's valid visual length before cross-attention.
        target_length = x_visual.shape[-1]
        x_audio, aligned_audio_mask = self._adaptive_align_audio_to_video(
            x_audio,
            audio_stage1_lengths,
            video_lengths.to(x_audio.device),
            target_length,
        )
        attention_mask = ~(video_mask & aligned_audio_mask)

        # Modality dropout — forces each encoder to be independently capable
        if self.training:
            audio_mask  = (torch.rand(x_audio.size(0),  1, 1, device=x_audio.device)  > 0.15).float()
            visual_mask = (torch.rand(x_visual.size(0), 1, 1, device=x_visual.device) > 0.15).float()
            x_audio  = x_audio  * audio_mask
            x_visual = x_visual * visual_mask

        # Eval-time single-modality ablation. Zeroing happens at the same point as
        # modality dropout so the ablated stream matches what training already saw.
        ablate = getattr(self, 'ablate_modality', 'none')
        if ablate == 'audio_only':
            x_visual = torch.zeros_like(x_visual)
        elif ablate == 'video_only':
            x_audio = torch.zeros_like(x_audio)
        elif ablate != 'none':
            raise ValueError(
                f'Unsupported ablate_modality "{ablate}". Expected none, audio_only, or video_only.'
            )

        proj_x_a = x_audio.permute(0, 2, 1)
        proj_x_v = x_visual.permute(0, 2, 1)

        h_av = self.av1(proj_x_v, proj_x_a, key_padding_mask=attention_mask, query_padding_mask=attention_mask)
        h_va = self.va1(proj_x_a, proj_x_v, key_padding_mask=attention_mask, query_padding_mask=attention_mask)

        h_av = h_av.permute(0, 2, 1)
        h_va = h_va.permute(0, 2, 1)

        x_audio  = h_av + x_audio
        x_visual = h_va + x_visual

        x_audio  = self.audio_model.forward_stage2(x_audio)
        x_audio  = self.audio_feature_gate(x_audio)
        x_visual = self.visual_model.forward_stage2(x_visual)
        audio_stage2_mask = self._downsample_mask(aligned_audio_mask, levels=2)
        x_audio = x_audio * audio_stage2_mask.unsqueeze(1).to(x_audio.dtype)
        x_visual = x_visual * video_mask.unsqueeze(1).to(x_visual.dtype)

        # (T, B, C) format required by PyTorch MultiheadAttention
        x_audio  = x_audio.permute(2, 0, 1)
        x_visual = x_visual.permute(2, 0, 1)

        # Cross-modal attention: audio queries video, video queries audio
        x_audio_attention,  _ = self.audioAttention(
            x_audio,
            x_visual,
            x_visual,
            key_padding_mask=~video_mask,
        )
        x_visual_attention, _ = self.visualAttention(
            x_visual,
            x_audio,
            x_audio,
            key_padding_mask=~audio_stage2_mask,
        )

        # Residual + dropout
        x_audio  = x_audio  + self.attn_dropout(x_audio_attention)
        x_visual = x_visual + self.attn_dropout(x_visual_attention)
        x_audio = x_audio * audio_stage2_mask.transpose(0, 1).unsqueeze(-1).to(x_audio.dtype)
        x_visual = x_visual * video_mask.transpose(0, 1).unsqueeze(-1).to(x_visual.dtype)

        x_audio_ca  = x_audio.permute(1, 0, 2)   # (B, T, C)
        x_visual_ca = x_visual.permute(1, 0, 2)

        x_audio_final  = self.audioCrossAttention(
            xk=x_visual_ca,
            xq=x_audio_ca,
            key_padding_mask=~video_mask,
            query_padding_mask=~audio_stage2_mask,
        )
        x_visual_final = self.visualCrossAttention(
            xk=x_audio_ca,
            xq=x_visual_ca,
            key_padding_mask=~audio_stage2_mask,
            query_padding_mask=~video_mask,
        )

        # Learned attention pooling over temporal dimension
        audio_pooled = self.attn_pool_audio(x_audio_final, mask=audio_stage2_mask)
        video_pooled = self.attn_pool_video(x_visual_final, mask=video_mask)

        av_pair = torch.cat((audio_pooled, video_pooled), dim=-1)
        if self.behavior or self.text_fusion:
            extras = [av_pair]
            if self.behavior:
                extras.append(self._behavior_fusion(av_pair, behavior_feats, behavior_present))
            if self.text_fusion:
                extras.append(self._text_fusion(av_pair, behavior_feats, behavior_present))
            return self.classifier_fused(torch.cat(extras, dim=-1))
        x = av_pair
        if self.late_text_fusion:
            av_context = self.av_context(x)
            refined_context = self.text_addon(av_context, text_tokens=text_tokens, text_mask=text_mask)
            x = torch.cat((av_context, refined_context), dim=-1)
        return self.classifier_1(x)

    def _behavior_fusion(self, av_pair, behavior_feats, behavior_present):
        """Behavior context (AV summary attends behavior tokens) + direct AU skip.

        Absent samples get a learnable missing-modality token and a zeroed skip.
        Returns (B, e_dim + behavior_skip_dim).
        """
        batch = av_pair.shape[0]
        device = av_pair.device
        if behavior_feats is None:
            ctx = self.behavior_missing.to(device=device, dtype=av_pair.dtype).expand(batch, -1)
            skip = torch.zeros(batch, self.behavior_skip_dim, device=device, dtype=av_pair.dtype)
            return torch.cat((ctx, skip), dim=-1)

        behavior_feats = behavior_feats.to(device=device, dtype=av_pair.dtype)
        if behavior_present is None:
            behavior_present = torch.ones(batch, dtype=torch.bool, device=device)
        else:
            behavior_present = behavior_present.to(device).bool().view(-1)

        enc = self.behavior_encoder(behavior_feats, present=behavior_present)
        tokens = enc['tokens']                              # (B, T, e_dim)
        query = self.behavior_av_proj(av_pair).unsqueeze(1)  # (B, 1, e_dim)

        # AV summary queries the behavior token sequence. Behavior tokens are a
        # dense 15-frame stream (no per-token padding), so no key mask is needed;
        # whole-clip absence is handled by the missing-modality token below.
        ctx, _ = self.behaviorCrossAttention(query, tokens, tokens)
        ctx = ctx.squeeze(1)                               # (B, e_dim)
        if (~behavior_present).any():
            miss = self.behavior_missing.to(device=device, dtype=av_pair.dtype).expand(batch, -1)
            ctx = torch.where(behavior_present.view(batch, 1), ctx, miss)

        skip = self.behavior_skip(behavior_feats.mean(dim=1))          # (B, skip_dim)
        skip = skip * behavior_present.view(batch, 1).to(skip.dtype)
        return torch.cat((ctx, skip), dim=-1)

    def _captions_from_feats(self, behavior_feats, behavior_present):
        """Turn each clip's mean AU/gaze/pose vector into a unified text stream.

        Text is currently the behavior caption (chat can be threaded in later);
        absent behavior -> empty stream -> missing token downstream.
        """
        from models.behavior_features import caption_stream
        from models.behavior_captioner import build_unified_stream

        feats = behavior_feats.detach().cpu().float().numpy()
        present = None if behavior_present is None else behavior_present.detach().cpu()
        streams = []
        for i in range(feats.shape[0]):
            if present is not None and not bool(present[i]):
                streams.append([])
                continue
            caption = caption_stream(feats[i].mean(axis=0))
            streams.append(build_unified_stream(chat="", behavior_caption=caption))
        return streams

    def _text_fusion(self, av_pair, behavior_feats, behavior_present):
        """Behavior-caption text -> frozen sentence encoder -> AV summary cross-attends it.

        Absent behavior (or an empty caption) falls back to a learnable missing
        token. Returns (B, e_dim).
        """
        batch = av_pair.shape[0]
        device = av_pair.device
        if behavior_feats is None:
            return self.text_missing.to(device=device, dtype=av_pair.dtype).expand(batch, -1)

        streams = self._captions_from_feats(behavior_feats, behavior_present)
        tokens, mask = self.text_encoder(streams, device=device)   # (B, M, e_dim), (B, M)
        tokens = tokens.to(dtype=av_pair.dtype)
        query = self.text_av_proj(av_pair).unsqueeze(1)            # (B, 1, e_dim)

        empty = ~mask.any(dim=1)                                   # rows with no usable text
        safe_mask = (~mask).clone()
        safe_mask[empty] = False                                  # avoid all-pad rows (NaN)
        ctx, _ = self.textCrossAttention(query, tokens, tokens, key_padding_mask=safe_mask)
        ctx = ctx.squeeze(1)
        if empty.any():
            miss = self.text_missing.to(device=device, dtype=av_pair.dtype).expand(batch, -1)
            ctx = torch.where(empty.view(batch, 1), miss, ctx)
        return ctx

    def forward_feature_3_legacy(self, x_audio, x_visual):
        x_audio = self.audio_model.forward_stage1(x_audio)
        x_visual = self.visual_model.forward_features(x_visual)
        x_visual = self.visual_model.forward_stage1(x_visual)

        proj_x_a = x_audio.permute(0, 2, 1)
        proj_x_v = x_visual.permute(0, 2, 1)

        h_av = self.av1(proj_x_v, proj_x_a)
        h_va = self.va1(proj_x_a, proj_x_v)

        h_av = h_av.permute(0, 2, 1)
        h_va = h_va.permute(0, 2, 1)

        x_audio = h_av + x_audio
        x_visual = h_va + x_visual

        x_audio = self.audio_model.forward_stage2(x_audio)
        x_audio = self.audio_feature_gate(x_audio)
        x_visual = self.visual_model.forward_stage2(x_visual)

        x_audio = x_audio.permute(2, 0, 1)
        x_visual = x_visual.permute(2, 0, 1)

        x_audio_attention, _ = self.audioAttention(x_audio, x_audio, x_audio)
        x_visual_attention, _ = self.visualAttention(x_visual, x_visual, x_visual)

        x_audio_attention = x_audio_attention.permute(1, 2, 0)
        x_visual_attention = x_visual_attention.permute(1, 2, 0)

        x_audio_ca = x_audio_attention.permute(0, 2, 1)
        x_visual_ca = x_visual_attention.permute(0, 2, 1)

        x_audio_final = self.audioCrossAttention(xk=x_visual_ca, xq=x_audio_ca)
        x_visual_final = self.visualCrossAttention(xk=x_audio_ca, xq=x_visual_ca)

        audio_pooled = x_audio_final.max(dim=1).values
        video_pooled = x_visual_final.max(dim=1).values

        x = torch.cat((audio_pooled, video_pooled), dim=-1)
        x1 = self.classifier_1(x)
        return x1
    
    def forward_feature_2(self, x_audio, x_visual):
        x_audio = self.audio_model.forward_stage1(x_audio)
        x_visual = self.visual_model.forward_features(x_visual)
        x_visual = self.visual_model.forward_stage1(x_visual)

        proj_x_a = x_audio.permute(0, 2, 1)
        proj_x_v = x_visual.permute(0, 2, 1)

        h_av, _ = self.av1(proj_x_v, proj_x_a)
        h_va, _ = self.va1(proj_x_a, proj_x_v)

        h_av = h_av.permute(0, 2, 1)
        h_va = h_va.permute(0, 2, 1)

        x_audio  = x_audio  + h_av
        x_visual = x_visual + h_va

        x_audio  = self.audio_model.forward_stage2(x_audio)
        x_audio  = self.audio_feature_gate(x_audio)
        x_visual = self.visual_model.forward_stage2(x_visual)

        audio_pooled = x_audio.max(dim=-1).values
        video_pooled = x_visual.max(dim=-1).values

        x  = torch.cat((audio_pooled, video_pooled), dim=-1)
        x1 = self.classifier_1(x)
        return x1

    def forward_transformer(self, x_audio, x_visual):
        x_audio = self.audio_model.forward_stage1(x_audio)
        proj_x_a = self.audio_model.forward_stage2(x_audio)
        proj_x_a = self.audio_feature_gate(proj_x_a)
       
        x_visual = self.visual_model.forward_features(x_visual) 
        x_visual = self.visual_model.forward_stage1(x_visual)
        proj_x_v = self.visual_model.forward_stage2(x_visual)

        proj_x_a = proj_x_a.permute(0, 2, 1)
        proj_x_v = proj_x_v.permute(0, 2, 1)
        h_av = self.av(proj_x_v, proj_x_a)
        h_va = self.va(proj_x_a, proj_x_v)
       
        audio_pooled = h_av.max(dim=1).values
        video_pooled = h_va.max(dim=1).values

        x = torch.cat((audio_pooled, video_pooled), dim=-1)  
        x1 = self.classifier_1(x)
        return x1
 
