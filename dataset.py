from datasets.ravdess import RAVDESS
from datasets.mosei import CMU_MOSEI
from datasets.cremad import CREMA_D

DATASET_REGISTRY = {
    'RAVDESS': RAVDESS,
    'CMU_MOSEI': CMU_MOSEI,
    'CREMA_D': CREMA_D
}

def get_training_set(opt, spatial_transform=None, audio_transform=None):
    assert opt.dataset in DATASET_REGISTRY, print('Unsupported dataset: {}'.format(opt.dataset))

    dataset_class = DATASET_REGISTRY[opt.dataset]
    training_data = dataset_class(
        opt.annotation_path,
        'training',
        spatial_transform=spatial_transform, data_type='audiovisual', audio_transform=audio_transform,
        data_root=opt.data_root)
    return training_data


def get_validation_set(opt, spatial_transform=None, audio_transform=None):
    assert opt.dataset in DATASET_REGISTRY, print('Unsupported dataset: {}'.format(opt.dataset))

    dataset_class = DATASET_REGISTRY[opt.dataset]
    validation_data = dataset_class(
        opt.annotation_path,
        'validation',
        spatial_transform=spatial_transform, data_type = 'audiovisual', audio_transform=audio_transform,
        data_root=opt.data_root)
    return validation_data

def get_test_set(opt, spatial_transform=None, audio_transform=None):
    assert opt.dataset in DATASET_REGISTRY, print('Unsupported dataset: {}'.format(opt.dataset))
    assert opt.test_subset in ['val', 'test']

    if opt.test_subset == 'val':
        subset = 'validation'
    elif opt.test_subset == 'test':
        subset = 'testing'
    
    dataset_class = DATASET_REGISTRY[opt.dataset]
    test_data = dataset_class(
        opt.annotation_path,
        subset,
        spatial_transform=spatial_transform, data_type='audiovisual',audio_transform=audio_transform,
        data_root=opt.data_root)
    return test_data
