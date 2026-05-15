from datasets.ravdess import RAVDESS

DATASET_REGISTRY = {
    'RAVDESS': RAVDESS,
}


def get_training_set(opt, spatial_transform=None, audio_transform=None):
    assert opt.dataset in DATASET_REGISTRY, f'Unsupported dataset: {opt.dataset}'
    return DATASET_REGISTRY[opt.dataset](
        opt.annotation_path, 'training',
        spatial_transform=spatial_transform,
        data_type='audiovisual',
        audio_transform=audio_transform,
        data_root=opt.data_root,
        audio_features=getattr(opt, 'audio_features', 'mel'),
    )


def get_validation_set(opt, spatial_transform=None, audio_transform=None):
    assert opt.dataset in DATASET_REGISTRY, f'Unsupported dataset: {opt.dataset}'
    return DATASET_REGISTRY[opt.dataset](
        opt.annotation_path, 'validation',
        spatial_transform=spatial_transform,
        data_type='audiovisual',
        audio_transform=audio_transform,
        data_root=opt.data_root,
        audio_features=getattr(opt, 'audio_features', 'mel'),
    )


def get_test_set(opt, spatial_transform=None, audio_transform=None):
    assert opt.dataset in DATASET_REGISTRY, f'Unsupported dataset: {opt.dataset}'
    assert opt.test_subset in ['val', 'test'], f'Invalid test_subset: {opt.test_subset}'
    subset = 'validation' if opt.test_subset == 'val' else 'testing'
    return DATASET_REGISTRY[opt.dataset](
        opt.annotation_path, subset,
        spatial_transform=spatial_transform,
        data_type='audiovisual',
        audio_transform=audio_transform,
        data_root=opt.data_root,
        audio_features=getattr(opt, 'audio_features', 'mel'),
    )
