# -*- coding: utf-8 -*-
"""Bootstrap EngageNet dataset loader for the current AVT-CA pipeline."""

import csv
import json
import os

import numpy as np
import torch
import torch.utils.data as data

from datasets.ravdess import (
    _pil_to_tensor,
    get_default_video_loader,
    get_mel,
    get_mfccs,
    load_audio,
)


def make_dataset(subset, annotation_path):
    dataset = []
    with open(annotation_path, "r", newline="") as handle:
        reader = csv.reader(handle, delimiter=";")
        for row in reader:
            if not row:
                continue
            if len(row) < 4:
                raise ValueError(f"Malformed EngageNet annotation row: {row}")
            video_path, audio_path, label, split = row[:4]
            if split != subset:
                continue
            dataset.append(
                {
                    "video_path": video_path,
                    "audio_path": audio_path,
                    "label": int(label),
                    "text": row[4] if len(row) > 4 else "",
                    "subject": row[5] if len(row) > 5 else "",
                }
            )
    return dataset


class ENGAGENET(data.Dataset):
    def __init__(
        self,
        annotation_path,
        subset,
        spatial_transform=None,
        get_loader=get_default_video_loader,
        data_type="audiovisual",
        audio_transform=None,
        audio_feature_transform=None,
        data_root="",
        audio_features="mfcc",
        target_frames=None,
        frame_sampling="uniform",
        audio_target_secs=None,
        behavior=False,
        behavior_dir=None,
        behavior_baselines=None,
        behavior_num_frames=None,
    ):
        del data_root
        self.data = make_dataset(subset, annotation_path)
        self.spatial_transform = spatial_transform
        self.audio_transform = audio_transform
        self.audio_feature_transform = audio_feature_transform
        self.loader = get_loader(target_frames=target_frames, frame_sampling=frame_sampling)
        self.data_type = data_type
        self.audio_features = audio_features
        self.audio_target_secs = audio_target_secs

        self.behavior = behavior
        self.behavior_dir = behavior_dir
        self._behavior = None
        self._baselines = {}
        if behavior:
            from models.behavior_features import BehaviorFeatures

            frames = behavior_num_frames or target_frames or 15
            self._behavior = BehaviorFeatures(num_frames=frames)
            if behavior_baselines:
                with open(behavior_baselines) as handle:
                    self._baselines = {
                        key: np.asarray(vec, dtype=np.float32)
                        for key, vec in json.load(handle).items()
                    }

    def _behavior_for(self, index):
        raw = None
        if self.behavior_dir:
            stem = os.path.splitext(os.path.basename(self.data[index]["video_path"]))[0]
            npy_path = os.path.join(self.behavior_dir, f"{stem}.npy")
            if os.path.isfile(npy_path):
                raw = np.load(npy_path)
        baseline = self._baselines.get(self.data[index].get("subject", ""))
        return self._behavior.process(raw, baseline=baseline)

    def __getitem__(self, index):
        target = self.data[index]["label"]

        if self.data_type == "video" or self.data_type == "audiovisual":
            path = self.data[index]["video_path"]
            clip = self.loader(path)

            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]
            else:
                clip = [_pil_to_tensor(img) for img in clip]
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

            if self.data_type == "video":
                return clip, target

        if self.data_type == "audio" or self.data_type == "audiovisual":
            path = self.data[index]["audio_path"]
            y, sr = load_audio(path, sr=22050, target_secs=self.audio_target_secs)

            if self.audio_transform is not None:
                self.audio_transform.randomize_parameters()
                y = self.audio_transform(y)

            if self.audio_features == "mel":
                audio_features = get_mel(y, sr, n_mels=64)
            else:
                audio_features = get_mfccs(y, sr)

            if self.audio_feature_transform is not None:
                self.audio_feature_transform.randomize_parameters()
                audio_features = self.audio_feature_transform(audio_features)

            if self.data_type == "audio":
                return audio_features, target

        if self.data_type == "audiovisual":
            audio_features = torch.as_tensor(audio_features, dtype=torch.float32)
            clip = torch.as_tensor(clip, dtype=torch.float32).permute(1, 0, 2, 3)
            sample = (
                audio_features,
                clip,
                target,
                int(audio_features.shape[-1]),
                int(clip.shape[0]),
                self.data[index].get("text", ""),
            )
            if self.behavior:
                beh = self._behavior_for(index)
                sample = sample + (beh["features"], beh["present"])
            return sample

    def __len__(self):
        return len(self.data)
