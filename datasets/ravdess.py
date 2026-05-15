# -*- coding: utf-8 -*-
"""
This code is base on https://github.com/okankop/Efficient-3DCNNs
"""

import os
import torch
import torch.utils.data as data
from PIL import Image
import functools
import numpy as np
import librosa
import cv2


def video_loader(video_dir_path):
    if video_dir_path.endswith('.npy'):
        video = np.load(video_dir_path)
        video_data = []
        for i in range(np.shape(video)[0]):
            video_data.append(Image.fromarray(video[i, :, :, :]))
        return video_data

    if video_dir_path.endswith('.mp4'):
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
            frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(target_frames)]
        elif len(frames) >= target_frames:
            indices = np.linspace(0, len(frames) - 1, num=target_frames, dtype=int)
            frames = [frames[idx] for idx in indices]
        else:
            frames.extend([np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(target_frames - len(frames))])

        return [Image.fromarray(frame) for frame in frames]

    raise ValueError('Unsupported video format: {}'.format(video_dir_path))

def get_default_video_loader():
    return functools.partial(video_loader)

def load_audio(audiofile, sr):
    audios, sr = librosa.core.load(audiofile, sr=sr)
    target_length = int(sr * 3.6)
    if len(audios) < target_length:
        audios = np.pad(audios, (0, target_length - len(audios)))
    else:
        remain = len(audios) - target_length
        audios = audios[remain // 2:len(audios) - (remain - remain // 2)]
    return audios, sr

def get_mfccs(y, sr):
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)

def get_mel(y, sr, n_mels=64):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    return librosa.power_to_db(mel, ref=np.max)

def _normalize_path(path):
    return path.strip().replace('\\', '/')


def _extract_relative_path(path):
    normalized_path = _normalize_path(path)
    marker = 'RAVDESS/'
    if marker in normalized_path:
        return normalized_path.split(marker, 1)[1]

    parts = [part for part in normalized_path.split('/') if part]
    if len(parts) >= 2:
        return '/'.join(parts[-2:])
    return normalized_path


def _is_ravdess_root(path):
    if not path or not os.path.isdir(path):
        return False

    try:
        entries = os.listdir(path)
    except OSError:
        return False

    return any(entry.upper().startswith('ACTOR') for entry in entries)


def _candidate_roots(annotation_path, data_root):
    annotation_dir = os.path.dirname(os.path.abspath(annotation_path))
    # annotation is at <project>/preprocessing/ravdess/annotations.txt
    # so go up two levels to reach project root
    preprocessing_dir = os.path.dirname(annotation_dir)
    project_root = os.path.dirname(preprocessing_dir)

    candidates = []
    for candidate in [
        data_root,
        os.environ.get('RAVDESS_ROOT'),
        os.path.join(project_root, 'datasets', 'RAVDESS'),
        os.path.join(project_root, 'RAVDESS'),
        os.path.join(annotation_dir, 'RAVDESS'),
        os.path.expanduser('~/RAVDESS'),
        '/scratch/RAVDESS',
        '/data/RAVDESS',
        '/datasets/RAVDESS',
    ]:
        if not candidate:
            continue
        normalized_candidate = os.path.abspath(os.path.expanduser(candidate))
        if normalized_candidate not in candidates:
            candidates.append(normalized_candidate)
    return candidates


def _find_data_root(annotation_path, data_root):
    candidates = _candidate_roots(annotation_path, data_root)
    for candidate in candidates:
        if _is_ravdess_root(candidate):
            return candidate, candidates
    return None, candidates


def _resolve_sample_path(path, data_root):
    if os.path.isfile(path):
        return path

    normalized_path = _normalize_path(path)
    if os.path.isfile(normalized_path):
        return normalized_path

    if data_root:
        relative_path = _extract_relative_path(path)
        candidate_path = os.path.join(data_root, *relative_path.split('/'))
        if os.path.isfile(candidate_path):
            return candidate_path

        raw_relative_path = relative_path
        if raw_relative_path.endswith('_facecroppad.npy'):
            raw_relative_path = raw_relative_path.replace('_facecroppad.npy', '.mp4')
        elif raw_relative_path.endswith('_croppad.wav'):
            raw_relative_path = raw_relative_path.replace('_croppad.wav', '.wav')

        raw_candidate_path = os.path.join(data_root, *raw_relative_path.split('/'))
        if os.path.isfile(raw_candidate_path):
            return raw_candidate_path

    return None


def make_dataset(subset, annotation_path, data_root=''):
    resolved_data_root, attempted_roots = _find_data_root(annotation_path, data_root)

    with open(annotation_path, 'r') as f:
        annots = f.readlines()
        
    dataset = []
    for line in annots:
        line = line.strip()
        if not line:
            continue

        filename, audiofilename, label, trainvaltest = line.split(';')
        if trainvaltest.rstrip() != subset:
            continue

        resolved_video_path = _resolve_sample_path(filename, resolved_data_root)
        resolved_audio_path = _resolve_sample_path(audiofilename, resolved_data_root)

        if resolved_video_path is None or resolved_audio_path is None:
            attempted_locations = ', '.join(attempted_roots) if attempted_roots else '<none>'
            raise FileNotFoundError(
                'Could not resolve RAVDESS sample paths from annotations. '
                'Set --data_root or RAVDESS_ROOT to the directory that contains ACTORxx folders. '
                'Example unresolved video path: {}. '
                'Example unresolved audio path: {}. '
                'Attempted dataset roots: {}.'.format(
                    filename, audiofilename, attempted_locations))
        
        sample = {'video_path': resolved_video_path,
                  'audio_path': resolved_audio_path,
                  'label': int(label)-1}
        dataset.append(sample)
    return dataset 
       

class RAVDESS(data.Dataset):
    def __init__(self,
                 annotation_path,
                 subset,
                 spatial_transform=None,
                 get_loader=get_default_video_loader,
                 data_type='audiovisual',
                 audio_transform=None,
                 data_root='',
                 audio_features='mfcc'):
        self.data = make_dataset(subset, annotation_path, data_root=data_root)
        self.spatial_transform = spatial_transform
        self.audio_transform = audio_transform
        self.loader = get_loader()
        self.data_type = data_type
        self.audio_features = audio_features

    def __getitem__(self, index):
        target = self.data[index]['label']
                

        if self.data_type == 'video' or self.data_type == 'audiovisual':        
            path = self.data[index]['video_path']
            clip = self.loader(path)
            
            if self.spatial_transform is not None:               
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]            
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3) 
            
            if self.data_type == 'video':
                return clip, target
            
        if self.data_type == 'audio' or self.data_type == 'audiovisual':
            path = self.data[index]['audio_path']
            y, sr = load_audio(path, sr=22050) 
            
            if self.audio_transform is not None:
                 self.audio_transform.randomize_parameters()
                 y = self.audio_transform(y)     
                 
            if self.audio_features == 'mel':
                audio_features = get_mel(y, sr, n_mels=64)
            else:
                audio_features = get_mfccs(y, sr)

            if self.data_type == 'audio':
                return audio_features, target
        if self.data_type == 'audiovisual':
            return audio_features, clip, target  

    def __len__(self):
        return len(self.data)
