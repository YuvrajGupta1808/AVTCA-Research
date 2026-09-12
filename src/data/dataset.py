from datasets.ravdess import RAVDESS
from datasets.cremad import CREMAD
from datasets.engagenet import ENGAGENET
from datasets.daisee import DAISEE

DATASET_REGISTRY = {
    'RAVDESS': RAVDESS,
    'CREMAD': CREMAD,
    'ENGAGENET': ENGAGENET,
    'DAISEE': DAISEE,
}

TEST_SUBSET_ALIASES = {
    'val': 'validation',
    'test': 'testing',
}


def resolve_test_subset_name(test_subset):
    if test_subset not in TEST_SUBSET_ALIASES:
        valid = ', '.join(sorted(TEST_SUBSET_ALIASES))
        raise ValueError(f'Invalid test_subset "{test_subset}". Expected one of: {valid}')
    return TEST_SUBSET_ALIASES[test_subset]


def build_dataset(opt, subset, spatial_transform=None, audio_transform=None, audio_feature_transform=None):
    if opt.dataset not in DATASET_REGISTRY:
        valid = ', '.join(sorted(DATASET_REGISTRY))
        raise ValueError(f'Unsupported dataset "{opt.dataset}". Expected one of: {valid}')
    use_dynamic_temporal = opt.dataset in {'ENGAGENET', 'DAISEE'} and getattr(opt, 'full_video_preprocessing', False)
    target_frames = None if use_dynamic_temporal else getattr(opt, 'sample_duration', 15)
    audio_target_secs = None if use_dynamic_temporal else 3.6
    kwargs = dict(
        spatial_transform=spatial_transform,
        data_type='audiovisual',
        audio_transform=audio_transform,
        audio_feature_transform=audio_feature_transform,
        data_root=opt.data_root,
        audio_features=getattr(opt, 'audio_features', 'mel'),
        target_frames=target_frames,
        frame_sampling=getattr(opt, 'frame_sampling', 'uniform'),
        audio_target_secs=audio_target_secs,
    )
    # Behavior features are engagement-only for now; RAVDESS/CREMAD loaders do not
    # accept these kwargs, so gate them on the dataset.
    if opt.dataset == 'ENGAGENET' and (getattr(opt, 'behavior', False) or getattr(opt, 'text_fusion', False)):
        kwargs.update(
            behavior=True,
            behavior_dir=getattr(opt, 'behavior_dir', None),
            behavior_baselines=getattr(opt, 'behavior_baselines', None),
            behavior_num_frames=resolve_behavior_frames(opt),
        )
    return DATASET_REGISTRY[opt.dataset](opt.annotation_path, subset, **kwargs)


def resolve_behavior_frames(opt):
    """Time steps for the per-clip OpenFace series.

    ``--behavior_frames 0`` (default) follows ``--max_video_frames`` so the
    behavior tokens span the whole clip on the same clock as the face frames
    (50 at 5 fps for a 10 s EngageNet clip). The collaborator branch hardcoded
    15, i.e. 1.5 fps against 50-frame video; that value is still reachable
    explicitly for reproducing the 2026-08-18 runs.
    """
    explicit = int(getattr(opt, 'behavior_frames', 0) or 0)
    if explicit > 0:
        return explicit
    cap = int(getattr(opt, 'max_video_frames', 0) or 0)
    return cap if cap > 0 else 15


def get_training_set(opt, spatial_transform=None, audio_transform=None, audio_feature_transform=None):
    return build_dataset(
        opt,
        'training',
        spatial_transform=spatial_transform,
        audio_transform=audio_transform,
        audio_feature_transform=audio_feature_transform,
    )


def get_validation_set(opt, spatial_transform=None, audio_transform=None, audio_feature_transform=None):
    return build_dataset(
        opt,
        'validation',
        spatial_transform=spatial_transform,
        audio_transform=audio_transform,
        audio_feature_transform=audio_feature_transform,
    )


def get_test_set(opt, spatial_transform=None, audio_transform=None, audio_feature_transform=None):
    subset = resolve_test_subset_name(opt.test_subset)
    return build_dataset(
        opt,
        subset,
        spatial_transform=spatial_transform,
        audio_transform=audio_transform,
        audio_feature_transform=audio_feature_transform,
    )
