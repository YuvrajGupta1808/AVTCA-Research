import os

import torch

from src.config.opts import parse_opts
from src.engine.checkpointing import load_state_dict_flexible
from src.engine.evaluation import canonical_evaluate_split, run_validation_epoch
from src.engine.runtime import (
    build_criterion,
    build_training_components,
    build_validation_components,
    persist_run_options,
    prepare_run_options,
    print_runtime_summary,
)
from src.engine.train import train_epoch
from src.models.factory import generate_model
from src.utils.common import adjust_learning_rate, save_checkpoint


def main():
    opt = parse_opts()
    opt = prepare_run_options(opt)
    persist_run_options(opt)

    torch.manual_seed(opt.manual_seed)
    model, parameters = generate_model(opt)
    print_runtime_summary(opt, model)

    if not opt.resume_path and not (opt.no_train and opt.no_val and opt.test):
        checkpoint_path = os.path.join(opt.result_path, 'model.pth')
        if os.path.isfile(checkpoint_path):
            load_state_dict_flexible(model, checkpoint_path, map_location=torch.device(opt.device))
            print('Loaded model weights from {}'.format(checkpoint_path))
        else:
            print('No existing model weights found at {}. Starting from initialized weights.'.format(checkpoint_path))

    criterion = build_criterion(opt)

    optimizer = None
    if not opt.no_train:
        training_data, train_loader, train_logger, train_batch_logger, optimizer, _scheduler = build_training_components(
            opt, parameters
        )
        print(training_data)

    if not opt.no_val:
        _validation_data, val_loader, val_logger = build_validation_components(opt)

    best_prec1 = 0
    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = load_state_dict_flexible(model, opt.resume_path, map_location=torch.device(opt.device))
        assert opt.arch == checkpoint['arch']
        best_prec1 = checkpoint['best_prec1']
        opt.begin_epoch = checkpoint['epoch']

    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            adjust_learning_rate(optimizer, i, opt)
            train_epoch(i, train_loader, model, criterion, optimizer, opt, train_logger, train_batch_logger)
            state = {
                'epoch': i,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1,
            }
            save_checkpoint(state, False, opt)

        if not opt.no_val:
            validation_loss, prec1 = run_validation_epoch(i, val_loader, model, criterion, opt, val_logger)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            state = {
                'epoch': i,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1,
            }
            save_checkpoint(state, is_best, opt)

    if opt.test:
        checkpoint_path = opt.checkpoint_path or os.path.join(opt.result_path, f'{opt.store_name}_best.pth')
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f'Explicit test checkpoint not found: {checkpoint_path}. '
                'Pass --checkpoint_path or ensure the best checkpoint exists.'
            )
        load_state_dict_flexible(model, checkpoint_path, map_location=torch.device(opt.device))
        metrics, _, checkpoint_info, artifact_paths = canonical_evaluate_split(
            opt=opt,
            model=model,
            criterion=criterion,
            checkpoint_path=checkpoint_path,
            split_alias=opt.test_subset,
            epoch=10000,
            logger=None,
            status='verified',
            write_legacy_test_files=True,
        )
        print(
            'Canonical test evaluation complete: '
            f'checkpoint={checkpoint_info["filename"]} '
            f'top1={metrics["top1_accuracy"]:.4f} '
            f'artifact={artifact_paths["json"]}'
        )


if __name__ == '__main__':
    main()
