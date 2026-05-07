import math

import torch
from torch import nn
import torch.nn.functional as F


class AudioTokenizer(nn.Module):
    """Convert audio features or waveform-like tensors into [B, T_a, D] tokens."""

    def __init__(self, input_channels=10, embed_dim=256, hidden_dim=128, max_audio_tokens=128, dropout=0.2):
        super().__init__()
        self.input_channels = input_channels
        self.max_audio_tokens = max_audio_tokens
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.project = nn.Linear(hidden_dim, embed_dim)

    def _adapt_channels(self, audio):
        channels = audio.size(1)
        if channels == self.input_channels:
            return audio
        if channels < self.input_channels:
            repeats = math.ceil(self.input_channels / channels)
            return audio.repeat(1, repeats, 1)[:, :self.input_channels]

        # Pool the feature/channel axis while preserving the temporal axis.
        audio = audio.transpose(1, 2)
        audio = F.adaptive_avg_pool1d(audio, self.input_channels)
        return audio.transpose(1, 2)

    def _channels_first(self, audio):
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        if audio.dim() != 3:
            raise ValueError('audio must have shape [B, C, T], [B, T, C], or [B, T]')

        # Existing RAVDESS MFCCs are [B, 10, T]. Waveforms are commonly [B, 1, T].
        # If the last dimension is the smaller feature/channel axis, convert [B, T, C].
        if audio.size(2) == self.input_channels and audio.size(1) != self.input_channels:
            audio = audio.transpose(1, 2)
        elif audio.size(1) > audio.size(2) and audio.size(2) <= 64:
            audio = audio.transpose(1, 2)
        return self._adapt_channels(audio.float())

    def forward(self, audio):
        x = self._channels_first(audio)
        x = self.stem(x)
        x = F.adaptive_avg_pool1d(x, self.max_audio_tokens)
        x = x.transpose(1, 2).contiguous()
        return self.project(x)


class ChannelSpatialRefinement(nn.Module):
    """Lightweight channel + spatial refinement for visual frame features."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x):
        channel_refined = x * self.channel(x)
        avg_map = channel_refined.mean(dim=1, keepdim=True)
        max_map = channel_refined.amax(dim=1, keepdim=True)
        spatial_refined = channel_refined * self.spatial(torch.cat([avg_map, max_map], dim=1))
        return spatial_refined + self.local(spatial_refined)


class VideoTokenizer(nn.Module):
    """Convert image, video, or precomputed visual features into [B, T_v, D] tokens."""

    def __init__(
        self,
        embed_dim=256,
        input_channels=3,
        hidden_dim=128,
        token_grid=4,
        max_video_tokens=64,
        dropout=0.2,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.token_grid = token_grid
        self.max_video_tokens = max_video_tokens
        self.frame_stem = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
        self.refine = ChannelSpatialRefinement(hidden_dim)
        self.project = nn.Linear(hidden_dim, embed_dim)
        self.feature_project = nn.Linear(embed_dim, embed_dim)

    def _adapt_frame_channels(self, frames):
        channels = frames.size(1)
        if channels == self.input_channels:
            return frames
        if channels < self.input_channels:
            repeats = math.ceil(self.input_channels / channels)
            return frames.repeat(1, repeats, 1, 1)[:, :self.input_channels]
        return frames.mean(dim=1, keepdim=True).repeat(1, self.input_channels, 1, 1)

    def _adapt_feature_dim(self, video):
        feature_dim = video.size(-1)
        expected = self.feature_project.in_features
        if feature_dim == expected:
            return video
        if feature_dim < expected:
            repeats = math.ceil(expected / feature_dim)
            return video.repeat(1, 1, repeats)[:, :, :expected]
        pooled = F.adaptive_avg_pool1d(video.reshape(-1, 1, feature_dim), expected)
        return pooled.reshape(video.size(0), video.size(1), expected)

    def _tokens_from_feature_sequence(self, video):
        if video.dim() != 3:
            raise ValueError('precomputed visual features must have shape [B, T, C]')
        return self.feature_project(self._adapt_feature_dim(video.float()))

    def _tokens_from_frames(self, frames, batch_size, frames_per_sample):
        # frames: [B*T, C, H, W]. Output before pooling: [B, T*G*G, D].
        x = self.frame_stem(self._adapt_frame_channels(frames.float()))
        x = self.refine(x)
        x = F.adaptive_avg_pool2d(x, (self.token_grid, self.token_grid))
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.project(x)
        x = x.reshape(batch_size, frames_per_sample * self.token_grid * self.token_grid, -1)

        if x.size(1) > self.max_video_tokens:
            x = F.adaptive_avg_pool1d(x.transpose(1, 2), self.max_video_tokens).transpose(1, 2)
        return x

    def forward(self, video):
        if video.dim() == 3:
            return self._tokens_from_feature_sequence(video)

        if video.dim() == 4:
            return self._tokens_from_frames(video, batch_size=video.size(0), frames_per_sample=1)

        if video.dim() == 5:
            if video.size(2) in (1, 3):
                # [B, T, C, H, W]
                batch_size, frames_per_sample = video.size(0), video.size(1)
                frames = video.reshape(batch_size * frames_per_sample, video.size(2), video.size(3), video.size(4))
            elif video.size(1) in (1, 3):
                # [B, C, T, H, W], as returned by the current RAVDESS loader.
                batch_size, frames_per_sample = video.size(0), video.size(2)
                frames = video.permute(0, 2, 1, 3, 4).contiguous()
                frames = frames.view(batch_size * frames_per_sample, video.size(1), video.size(3), video.size(4))
            else:
                raise ValueError('video sequence must be [B, T, C, H, W] or [B, C, T, H, W]')
            return self._tokens_from_frames(frames, batch_size, frames_per_sample)

        raise ValueError('video must have shape [B, C, H, W], [B, T, C, H, W], [B, C, T, H, W], or [B, T, D]')


class TokenFusionGate(nn.Module):
    """Token-level context exchange before the shared transformer."""

    def __init__(self, embed_dim, dropout=0.2):
        super().__init__()
        self.audio_context = nn.Linear(embed_dim, embed_dim)
        self.video_context = nn.Linear(embed_dim, embed_dim)
        self.audio_gate = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim), nn.Sigmoid())
        self.video_gate = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)

    def forward(self, audio_tokens, video_tokens):
        video_summary = video_tokens.mean(dim=1, keepdim=True)
        audio_summary = audio_tokens.mean(dim=1, keepdim=True)
        audio_context = self.audio_context(video_summary).expand_as(audio_tokens)
        video_context = self.video_context(audio_summary).expand_as(video_tokens)
        audio_gate = self.audio_gate(torch.cat([audio_tokens, audio_context], dim=-1))
        video_gate = self.video_gate(torch.cat([video_tokens, video_context], dim=-1))
        audio_tokens = audio_tokens + self.dropout(audio_gate * audio_context)
        video_tokens = video_tokens + self.dropout(video_gate * video_context)
        return audio_tokens, video_tokens


class AttentionPool(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.score = nn.Linear(embed_dim, 1)

    def forward(self, tokens):
        weights = torch.softmax(self.score(tokens), dim=1)
        return torch.sum(weights * tokens, dim=1)


class TokenFusionAVTCA(nn.Module):
    """Single-backbone audio-video token fusion model.

    The classifier consumes only fused transformer outputs. There are no separate
    audio/video logits and no late audio_pool + video_pool prediction path.
    """

    def __init__(
        self,
        num_classes,
        audio_in_channels=10,
        video_in_channels=3,
        embed_dim=256,
        depth=4,
        num_heads=4,
        fusion_tokens=4,
        dropout=0.2,
        attention_dropout=0.1,
        max_audio_tokens=128,
        max_video_tokens=64,
        token_grid=4,
        mlp_ratio=4,
        debug=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.fusion_tokens_count = fusion_tokens
        self.debug = debug

        self.audio_tokenizer = AudioTokenizer(
            input_channels=audio_in_channels,
            embed_dim=embed_dim,
            max_audio_tokens=max_audio_tokens,
            dropout=dropout,
        )
        self.video_tokenizer = VideoTokenizer(
            input_channels=video_in_channels,
            embed_dim=embed_dim,
            token_grid=token_grid,
            max_video_tokens=max_video_tokens,
            dropout=dropout,
        )
        self.token_fusion_gate = TokenFusionGate(embed_dim, dropout=dropout)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.fusion_tokens = nn.Parameter(torch.zeros(1, fusion_tokens, embed_dim))
        self.cls_modality_embedding = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.fusion_modality_embedding = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.audio_modality_embedding = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.video_modality_embedding = nn.Parameter(torch.zeros(1, 1, embed_dim))
        max_position_tokens = 1 + fusion_tokens + max_audio_tokens + max_video_tokens
        self.position_embedding = nn.Parameter(torch.zeros(1, max_position_tokens, embed_dim))
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        encoder_layer.self_attn.dropout = attention_dropout
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.final_norm = nn.LayerNorm(embed_dim)
        self.attention_pool = AttentionPool(embed_dim)
        self.head_norm = nn.LayerNorm(embed_dim * 3)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )
        self._init_parameters()

    def _init_parameters(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.fusion_tokens, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_modality_embedding, std=0.02)
        nn.init.trunc_normal_(self.fusion_modality_embedding, std=0.02)
        nn.init.trunc_normal_(self.audio_modality_embedding, std=0.02)
        nn.init.trunc_normal_(self.video_modality_embedding, std=0.02)

    def _position_slice(self, seq_len, device, dtype):
        pos = self.position_embedding
        if seq_len <= pos.size(1):
            return pos[:, :seq_len].to(device=device, dtype=dtype)
        pos = F.interpolate(pos.transpose(1, 2), size=seq_len, mode='linear', align_corners=False)
        return pos.transpose(1, 2).to(device=device, dtype=dtype)

    def build_fused_sequence(self, audio_tokens, video_tokens):
        batch_size = audio_tokens.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1) + self.cls_modality_embedding
        fusion = self.fusion_tokens.expand(batch_size, -1, -1) + self.fusion_modality_embedding
        audio_tokens = audio_tokens + self.audio_modality_embedding
        video_tokens = video_tokens + self.video_modality_embedding
        tokens = torch.cat([cls, fusion, audio_tokens, video_tokens], dim=1)
        tokens = tokens + self._position_slice(tokens.size(1), tokens.device, tokens.dtype)
        return self.input_dropout(tokens)

    def pool_fused_tokens(self, encoded):
        encoded = self.final_norm(encoded)
        cls_out = encoded[:, 0]
        fusion_out = encoded[:, 1:1 + self.fusion_tokens_count].mean(dim=1)
        pooled_out = self.attention_pool(encoded[:, 1:])
        return self.head_norm(torch.cat([cls_out, fusion_out, pooled_out], dim=-1))

    def forward(self, audio, video, return_attn=False):
        audio_tokens = self.audio_tokenizer(audio)
        video_tokens = self.video_tokenizer(video)
        audio_tokens, video_tokens = self.token_fusion_gate(audio_tokens, video_tokens)

        tokens = self.build_fused_sequence(audio_tokens, video_tokens)
        if self.debug:
            print('TokenFusionAVTCA tokens:', tokens.shape)
        encoded = self.transformer(tokens)
        logits = self.classifier(self.pool_fused_tokens(encoded))

        if return_attn:
            return logits, {'tokens': encoded}
        return logits
