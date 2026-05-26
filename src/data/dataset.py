from datasets.ravdess import RAVDESS
from datasets.cremad import CREMAD

DATASET_REGISTRY = {
    'RAVDESS': RAVDESS,
    'CREMAD': CREMAD,
}

TEST_SUBSET_ALIASES = {
    'val': 'validation',
    'test': 'testing',
}


def resolve_test_subset_name(test_subset):
    assert test_subset in TEST_SUBSET_ALIASES, f'Invalid test_subset: {test_subset}'
    return TEST_SUBSET_ALIASES[test_subset]


def build_dataset(opt, subset, spatial_transform=None, audio_transform=None):
    assert opt.dataset in DATASET_REGISTRY, f'Unsupported dataset: {opt.dataset}'
    return DATASET_REGISTRY[opt.dataset](
        opt.annotation_path, subset,
        spatial_transform=spatial_transform,
        data_type='audiovisual',
        audio_transform=audio_transform,
        data_root=opt.data_root,
        audio_features=getattr(opt, 'audio_features', 'mel'),
    )


def get_training_set(opt, spatial_transform=None, audio_transform=None):
    return build_dataset(opt, 'training', spatial_transform=spatial_transform, audio_transform=audio_transform)


def get_validation_set(opt, spatial_transform=None, audio_transform=None):
    return build_dataset(opt, 'validation', spatial_transform=spatial_transform, audio_transform=audio_transform)


def get_test_set(opt, spatial_transform=None, audio_transform=None):
    subset = resolve_test_subset_name(opt.test_subset)
    return build_dataset(opt, subset, spatial_transform=spatial_transform, audio_transform=audio_transform)
