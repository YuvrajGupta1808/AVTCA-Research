import math
import hashlib
import re

import torch

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")


def _hash_token(token, vocab_size):
    if vocab_size <= 1:
        return 0
    digest = hashlib.blake2b(token.encode('utf-8'), digest_size=8).digest()
    return (int.from_bytes(digest, byteorder='big') % (vocab_size - 1)) + 1


def _encode_text(text, max_text_tokens, text_vocab_size):
    if not isinstance(text, str):
        text = '' if text is None else str(text)
    tokens = TOKEN_PATTERN.findall(text.lower())
    if max_text_tokens and max_text_tokens > 0:
        tokens = tokens[:max_text_tokens]
    if not tokens:
        return torch.zeros(0, dtype=torch.long)
    ids = [_hash_token(token, text_vocab_size) for token in tokens]
    return torch.tensor(ids, dtype=torch.long)


def sample_temporal_indices(length, target_length, mode='uniform'):
    length = int(length)
    target_length = int(target_length)
    if length <= 0 or target_length <= 0:
        return []
    if length <= target_length:
        return list(range(length))
    if mode == 'random':
        start = int(torch.randint(0, length - target_length + 1, (1,)).item())
        return list(range(start, start + target_length))
    if mode == 'stride':
        stride = max(int(math.ceil(length / float(target_length))), 1)
        indices = list(range(0, length, stride))[:target_length]
        if len(indices) < target_length:
            indices.extend([length - 1] * (target_length - len(indices)))
        return indices
    return torch.linspace(0, length - 1, steps=target_length).round().long().tolist()


def maybe_temporal_subsample(sequence, max_length, mode='uniform'):
    if max_length is None or max_length <= 0:
        return sequence
    current_length = int(sequence.shape[0])
    if current_length <= max_length:
        return sequence
    indices = sample_temporal_indices(current_length, max_length, mode=mode)
    if isinstance(sequence, torch.Tensor):
        index_tensor = torch.tensor(indices, dtype=torch.long)
        return sequence.index_select(0, index_tensor)
    return sequence[indices]


def _slice_temporal(sequence, start, end):
    if isinstance(sequence, torch.Tensor):
        return sequence[start:end]
    return sequence[start:end]


def _random_synced_audio_video_crop(audio, video, max_video_frames, max_audio_steps):
    video_length = int(video.shape[0])
    if max_video_frames is None or max_video_frames <= 0 or video_length <= max_video_frames:
        return audio, video, False

    video_start = int(torch.randint(0, video_length - max_video_frames + 1, (1,)).item())
    video_end = video_start + int(max_video_frames)
    video = _slice_temporal(video, video_start, video_end)

    audio_length = int(audio.shape[-1])
    if audio_length > 0:
        audio_start = int(math.floor((video_start / float(video_length)) * audio_length))
        audio_end = int(math.ceil((video_end / float(video_length)) * audio_length))
        audio_end = min(max(audio_end, audio_start + 1), audio_length)
        audio_start = max(min(audio_start, audio_end - 1), 0)
        audio = audio[:, audio_start:audio_end]
        if max_audio_steps and max_audio_steps > 0:
            audio = maybe_temporal_subsample(audio.transpose(0, 1), max_audio_steps, mode='uniform').transpose(0, 1)

    return audio, video, True


def _pad_sequence(sequence, target_length, pad_value):
    current_length = int(sequence.shape[0])
    if current_length == target_length:
        return sequence
    pad_shape = (target_length - current_length,) + tuple(sequence.shape[1:])
    pad = torch.full(pad_shape, pad_value, dtype=sequence.dtype)
    return torch.cat([sequence, pad], dim=0)


def collate_variable_length_batch(
    batch,
    max_video_frames=0,
    max_audio_steps=0,
    frame_sampling='uniform',
    temporal_pad_value=0.0,
    max_text_tokens=32,
    text_vocab_size=4096,
):
    audio_list = []
    video_list = []
    targets = []
    audio_lengths = []
    video_lengths = []
    text_list = []
    behavior_list = []
    behavior_present = []
    has_text_field = False
    has_behavior_field = bool(batch) and len(batch[0]) >= 8

    for sample in batch:
        if len(sample) >= 6:
            audio_features, video_frames, target, audio_len, video_len, text = sample[:6]
            has_text_field = True
        else:
            audio_features, video_frames, target, audio_len, video_len = sample
            text = ''
        audio = torch.as_tensor(audio_features, dtype=torch.float32)
        video = torch.as_tensor(video_frames, dtype=torch.float32)

        if frame_sampling == 'random':
            audio, video, synced_crop = _random_synced_audio_video_crop(
                audio,
                video,
                max_video_frames=max_video_frames,
                max_audio_steps=max_audio_steps,
            )
            if not synced_crop and max_audio_steps and max_audio_steps > 0:
                audio = maybe_temporal_subsample(audio.transpose(0, 1), max_audio_steps, mode='random').transpose(0, 1)
        else:
            video = maybe_temporal_subsample(video, max_video_frames, mode=frame_sampling)
            if max_audio_steps and max_audio_steps > 0:
                audio = maybe_temporal_subsample(audio.transpose(0, 1), max_audio_steps, mode=frame_sampling).transpose(0, 1)

        audio_list.append(audio)
        video_list.append(video)
        targets.append(int(target))
        audio_lengths.append(int(audio.shape[-1]))
        video_lengths.append(int(video.shape[0]))
        if has_text_field:
            text_list.append(_encode_text(text, max_text_tokens=max_text_tokens, text_vocab_size=text_vocab_size))
        if has_behavior_field:
            behavior_list.append(torch.as_tensor(sample[6], dtype=torch.float32))
            behavior_present.append(bool(sample[7]))

    max_audio = max(audio_lengths) if audio_lengths else 0
    max_video = max(video_lengths) if video_lengths else 0

    padded_audio = []
    padded_video = []
    audio_masks = []
    video_masks = []

    for audio, video, audio_len, video_len in zip(audio_list, video_list, audio_lengths, video_lengths):
        audio_t = audio.transpose(0, 1)
        audio_t = _pad_sequence(audio_t, max_audio, temporal_pad_value)
        padded_audio.append(audio_t.transpose(0, 1))

        padded_video.append(_pad_sequence(video, max_video, temporal_pad_value))

        audio_mask = torch.zeros(max_audio, dtype=torch.bool)
        video_mask = torch.zeros(max_video, dtype=torch.bool)
        audio_mask[:audio_len] = True
        video_mask[:video_len] = True
        audio_masks.append(audio_mask)
        video_masks.append(video_mask)

    collated = (
        torch.stack(padded_audio, dim=0),
        torch.stack(padded_video, dim=0),
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(audio_lengths, dtype=torch.long),
        torch.tensor(video_lengths, dtype=torch.long),
        torch.stack(audio_masks, dim=0),
        torch.stack(video_masks, dim=0),
    )
    behavior_extra = ()
    if has_behavior_field:
        behavior_extra = (
            torch.stack(behavior_list, dim=0),
            torch.tensor(behavior_present, dtype=torch.bool),
        )

    if not has_text_field:
        return collated + behavior_extra

    max_text = max((text.shape[0] for text in text_list), default=0)
    max_text = max(max_text, 1)
    padded_text = []
    text_masks = []
    for text in text_list:
        text_mask = torch.zeros(max_text, dtype=torch.bool)
        text_mask[:text.shape[0]] = True
        text_masks.append(text_mask)
        if text.shape[0] < max_text:
            pad = torch.zeros(max_text - text.shape[0], dtype=torch.long)
            text = torch.cat([text, pad], dim=0)
        padded_text.append(text)

    return collated + (
        torch.stack(padded_text, dim=0),
        torch.stack(text_masks, dim=0),
    ) + behavior_extra
