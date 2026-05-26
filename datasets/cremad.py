# -*- coding: utf-8 -*-

import functools
import os

import cv2
import librosa
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


# ── video loading ────────────────────────────────────────────────────────────

def video_loader(video_dir_path):
    if video_dir_path.endswith('.npy'):
        video = np.load(video_dir_path)
        return [Image.fromarray(video[i]) for i in range(video.shape[0])]

    if video_dir_path.endswith('.flv'):
        cap = cv2.VideoCapture(video_dir_path)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        cap.release()

        target_frames = 15
        if not frames:
            frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * target_frames
        elif len(frames) >= target_frames:
            indices = np.linspace(0, len(frames) - 1, num=target_frames, dtype=int)
            frames = [frames[i] for i in indices]
        else:
            blank = np.zeros((224, 224, 3), dtype=np.uint8)
            frames += [blank] * (target_frames - len(frames))

        return [Image.fromarray(f) for f in frames]

    raise ValueError(f'Unsupported video format: {video_dir_path}')


def get_default_video_loader():
    return functools.partial(video_loader)


# ── audio loading ────────────────────────────────────────────────────────────

def load_audio(audiofile, sr=22050, target_secs=3.6):
    y, sr = librosa.core.load(audiofile, sr=sr)
    target_length = int(sr * target_secs)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        remain = len(y) - target_length
        y = y[remain // 2: len(y) - (remain - remain // 2)]
    return y, sr


def get_mfccs(y, sr):
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)


def get_mel(y, sr, n_mels=64):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    return librosa.power_to_db(mel, ref=np.max)


# ── path resolution ──────────────────────────────────────────────────────────

def _normalize_path(path):
    return path.strip().replace('\\', '/')


def _is_cremad_root(path):
    if not path or not os.path.isdir(path):
        return False
    try:
        entries = set(os.listdir(path))
    except OSError:
        return False
    return 'VideoFlash' in entries


def _candidate_roots(annotation_path, data_root):
    annotation_dir = os.path.dirname(os.path.abspath(annotation_path))
    preprocessing_dir = os.path.dirname(annotation_dir)
    project_root = os.path.dirname(preprocessing_dir)

    candidates = []
    for candidate in [
        data_root,
        os.environ.get('CREMAD_ROOT'),
        os.path.join(project_root, 'datasets', 'CREMAD'),
        os.path.join(project_root, 'CREMAD'),
        os.path.expanduser('~/CREMAD'),
        '/scratch/CREMAD',
        '/data/CREMAD',
        '/datasets/CREMAD',
    ]:
        if not candidate:
            continue
        c = os.path.abspath(os.path.expanduser(candidate))
        if c not in candidates:
            candidates.append(c)
    return candidates


def _find_data_root(annotation_path, data_root):
    candidates = _candidate_roots(annotation_path, data_root)
    for c in candidates:
        if _is_cremad_root(c):
            return c, candidates
    return None, candidates


def _resolve_sample_path(path, data_root):
    if os.path.isfile(path):
        return path

    normalized = _normalize_path(path)
    if os.path.isfile(normalized):
        return normalized

    if data_root:
        # Both video (.npy) and audio (_croppad.wav) live in VideoFlash/
        if 'VideoFlash/' in normalized:
            rel = 'VideoFlash/' + normalized.split('VideoFlash/', 1)[1]
            candidate = os.path.join(data_root, *rel.split('/'))
            if os.path.isfile(candidate):
                return candidate

            # fall back to raw file (npy→flv, croppad.wav→.wav not needed
            # since audio is extracted from flv, but handle missing croppad)
            raw_rel = rel
            if raw_rel.endswith('_facecroppad.npy'):
                raw_rel = raw_rel.replace('_facecroppad.npy', '.flv')
            raw_candidate = os.path.join(data_root, *raw_rel.split('/'))
            if os.path.isfile(raw_candidate):
                return raw_candidate

    return None


# ── dataset builder ──────────────────────────────────────────────────────────

def make_dataset(subset, annotation_path, data_root=''):
    resolved_root, attempted_roots = _find_data_root(annotation_path, data_root)

    with open(annotation_path, 'r') as f:
        lines = f.readlines()

    dataset = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        video_path, audio_path, label, split = line.split(';')
        if split.rstrip() != subset:
            continue

        rv = _resolve_sample_path(video_path, resolved_root)
        ra = _resolve_sample_path(audio_path, resolved_root)

        if rv is None or ra is None:
            attempted = ', '.join(attempted_roots) if attempted_roots else '<none>'
            raise FileNotFoundError(
                'Could not resolve CREMA-D sample paths from annotations. '
                'Set --data_root or CREMAD_ROOT to the directory containing VideoFlash/. '
                'Run extract_audios.py before create_annotations.py to generate _croppad.wav files. '
                f'Unresolved video: {video_path}. '
                f'Unresolved audio: {audio_path}. '
                f'Attempted roots: {attempted}.'
            )

        dataset.append({
            'video_path': rv,
            'audio_path': ra,
            'label': int(label) - 1,  # 0-indexed
        })

    return dataset


# ── dataset class ────────────────────────────────────────────────────────────

class CREMAD(data.Dataset):
    """CREMA-D: 6 emotion classes (anger, disgust, fear, happy, neutral, sad)."""

    def __init__(
        self,
        annotation_path,
        subset,
        spatial_transform=None,
        get_loader=get_default_video_loader,
        data_type='audiovisual',
        audio_transform=None,
        data_root='',
        audio_features='mfcc',
    ):
        self.data = make_dataset(subset, annotation_path, data_root=data_root)
        self.spatial_transform = spatial_transform
        self.audio_transform = audio_transform
        self.loader = get_loader()
        self.data_type = data_type
        self.audio_features = audio_features

    def __getitem__(self, index):
        target = self.data[index]['label']

        if self.data_type in ('video', 'audiovisual'):
            clip = self.loader(self.data[index]['video_path'])
            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
            if self.data_type == 'video':
                return clip, target

        if self.data_type in ('audio', 'audiovisual'):
            y, sr = load_audio(self.data[index]['audio_path'], sr=22050)
            if self.audio_transform is not None:
                self.audio_transform.randomize_parameters()
                y = self.audio_transform(y)
            if self.audio_features == 'mel':
                audio_features = get_mel(y, sr, n_mels=64)
            else:
                audio_features = get_mfccs(y, sr)
            if self.data_type == 'audio':
                return audio_features, target

        return audio_features, clip, target

    def __len__(self):
        return len(self.data)
