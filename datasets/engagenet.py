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


_MARLIN_SPLIT_DIRS = {
    "training": "MARLIN_Train",
    "validation": "MARLIN_Validation",
    "testing": "MARLIN_Test",
}


def _marlin_split_dir(subset, marlin_root):
    if subset not in _MARLIN_SPLIT_DIRS:
        raise ValueError(f"No MARLIN directory mapping for subset {subset!r}")
    directory = os.path.join(marlin_root, _MARLIN_SPLIT_DIRS[subset])
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"MARLIN feature directory not found: {directory}. "
            "Pass --marlin_root pointing at the EngageNet root that contains MARLIN_*."
        )
    return directory


def _resample_tokens(features, target_tokens):
    """Force a MARLIN sequence to exactly ``target_tokens`` frames.

    The shipped EngageNet MARLIN features are NOT length-consistent: the modal
    length is 9, but 592 training clips are 312 tokens and several hundred are
    1-8 (measured 2026-08-12 over all 11,311 files). Two consequences:

      * length 1-2 crashes training outright, because the audio stage-2 stack
        halves the sequence twice and 1 // 2 // 2 == 0;
      * mixing 9-token and 312-token clips in one batch means the model sees
        two completely different temporal granularities for the same task,
        which is the same defect class as the 15-vs-50 frame split documented
        in docs/plan.md 13.1.

    Longer sequences are uniformly subsampled; shorter ones repeat their last
    token. Both are deterministic, so validation and test stay reproducible.
    """
    length = int(features.shape[0])
    if length == target_tokens:
        return features
    if length > target_tokens:
        idx = torch.linspace(0, length - 1, steps=target_tokens).round().long()
        return features.index_select(0, idx)
    pad = features[-1:].expand(target_tokens - length, -1)
    return torch.cat([features, pad], dim=0)


def _load_marlin(video_path, marlin_dir, target_tokens=9):
    """Map an annotation video path to its MARLIN feature tensor.

    Annotations point at ``<split>/<stem>_facecroppad.npy``; MARLIN files are
    ``MARLIN_<Split>/<stem>.mp4.pt``.
    """
    stem = os.path.basename(video_path)
    for suffix in ("_facecroppad.npy", ".npy"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    path = os.path.join(marlin_dir, stem + ".mp4.pt")
    features = torch.load(path, map_location="cpu")
    if not isinstance(features, torch.Tensor):
        raise TypeError(f"MARLIN feature {path} is {type(features).__name__}, expected Tensor")
    if features.dim() != 2:
        raise ValueError(f"MARLIN feature {path} has shape {tuple(features.shape)}, expected (T, C)")
    features = features.to(torch.float32)
    if target_tokens and target_tokens > 0:
        features = _resample_tokens(features, int(target_tokens))
    return features


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
        visual_features="frames",
        marlin_root="",
        marlin_tokens=9,
        behavior=False,
        behavior_dir=None,
        behavior_baselines=None,
        behavior_num_frames=None,
    ):
        del data_root
        self.data = make_dataset(subset, annotation_path)
        # 'marlin' replaces the raw-frame visual stream with precomputed MARLIN
        # embeddings, one (T, 1024) tensor per clip. 1024 is exactly what the
        # visual conv1d_0 expects, so these feed straight into
        # forward_stage1_from_sequence and the EfficientFace backbone is unused.
        self.visual_features = visual_features
        self.marlin_root = marlin_root
        self.marlin_tokens = marlin_tokens
        if visual_features == "marlin":
            self._marlin_dir = _marlin_split_dir(subset, marlin_root)
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
            self._assert_behavior_present()

    # Annotation video_path points at the face-crop array
    # (subject_..._vid_0_4_facecroppad.npy) while extract_behavior.py names its
    # output from the source video stem (subject_..._vid_0_4.npy). Strip the
    # known crop suffixes so the two line up.
    _CROP_SUFFIXES = ("_facecroppad", "_croppad", "_facecrop")

    def _behavior_path(self, index):
        if not self.behavior_dir:
            return None
        stem = os.path.splitext(os.path.basename(self.data[index]["video_path"]))[0]
        candidates = [stem]
        for suffix in self._CROP_SUFFIXES:
            if stem.endswith(suffix):
                candidates.append(stem[: -len(suffix)])
                break
        for candidate in candidates:
            npy_path = os.path.join(self.behavior_dir, f"{candidate}.npy")
            if os.path.isfile(npy_path):
                return npy_path
        return None

    def _behavior_for(self, index):
        raw = None
        npy_path = self._behavior_path(index)
        if npy_path is not None:
            raw = np.load(npy_path)
        baseline = self._baselines.get(self.data[index].get("subject", ""))
        return self._behavior.process(raw, baseline=baseline)

    def _assert_behavior_present(self, sample_size=200):
        """Fail loudly when the behavior stream resolves to nothing.

        A missing .npy silently yields zeros with present=False, so a naming or
        path mistake trains on an empty modality and looks like a bad result
        rather than a broken run. Check a sample up front instead.
        """
        if not self.behavior_dir:
            raise ValueError(
                'behavior/text fusion is enabled but --behavior_dir is empty; '
                'every clip would get a zeroed behavior stream.'
            )
        n = min(sample_size, len(self.data))
        found = sum(self._behavior_path(i) is not None for i in range(n))
        if found == 0:
            example = os.path.basename(self.data[0]["video_path"]) if self.data else '?'
            raise FileNotFoundError(
                f'No behavior .npy resolved for any of the first {n} clips in '
                f'{self.behavior_dir!r} (e.g. {example}). Run '
                f'preprocessing/engagenet/extract_behavior.py first.'
            )
        if found < n:
            print(f'  [behavior] warning: {n - found}/{n} sampled clips have no '
                  f'behavior .npy; those train on a zeroed stream.')
        else:
            print(f'  [behavior] {found}/{n} sampled clips resolved OK '
                  f'(resampled to {self._behavior.num_frames} steps per clip).')

    def __getitem__(self, index):
        target = self.data[index]["label"]

        precomputed_visual = self.visual_features == "marlin"

        if self.data_type == "video" or self.data_type == "audiovisual":
            path = self.data[index]["video_path"]
            if precomputed_visual:
                # Already (T, C); no decoding, no spatial transform, no permute.
                clip = _load_marlin(path, self._marlin_dir, self.marlin_tokens)
                if self.data_type == "video":
                    return clip, target
            else:
                clip = self.loader(path)

            if not precomputed_visual:
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
            clip = torch.as_tensor(clip, dtype=torch.float32)
            if not precomputed_visual:
                # (C, T, H, W) -> (T, C, H, W). MARLIN is already (T, C).
                clip = clip.permute(1, 0, 2, 3)
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
