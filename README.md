# Multimodal-Emotion-Recognition-using-AVTCA

This repository implements a multimodal network for emotion recognition using the Audio-Video Transformer Fusion with Cross Attention (AVT-CA) model, as given in the paper [Multimodal Emotion Recognition using Audio-Video Transformer Fusion with Cross Attention](https://arxiv.org/pdf/2407.18552). The implementation supports the RAVDESS dataset, which includes speech and frontal face view data across 8 distinct emotions: 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, and 08 = surprised.

<p align="center">
<img src="https://github.com/shravan-18/AVTCA/blob/main/img/AVTCA.png" alt="drawing" height="70%"/>
</p>
<p align = "center">
AVT-CA Model Diagram
</p>

Feel free to play around with the code, and let us know if you have any questions or face any issues!

## Citation

If you use our work, please cite as:
```bibtex
@misc{AVTCA,
      title={Multimodal Emotion Recognition using Audio-Video Transformer Fusion with Cross Attention}, 
      author={Joe Dhanith P R and Shravan Venkatraman and Modigari Narendra and Vigya Sharma and Santhosh Malarvannan and Amir H. Gandomi},
      year={2024},
      eprint={2407.18552},
      archivePrefix={arXiv},
      primaryClass={cs.MM},
      url={https://arxiv.org/abs/2407.18552}, 
}
```

If you are referencing our work, please also cite the following related paper:

**Chumachenko, K., Iosifidis, A., & Gabbouj, M. (2022).** *Self-attention fusion for audiovisual emotion recognition with incomplete data*. arXiv. https://arxiv.org/abs/2201.11095

## References

This work incorporates EfficientFace, available at [EfficientFace GitHub repository](https://github.com/zengqunzhao/EfficientFace). Please cite the paper titled "Robust Lightweight Facial Expression Recognition Network with Label Distribution Training" if you use EfficientFace. We appreciate @zengqunzhao for providing both the implementation and the pretrained model for EfficientFace!

The training pipeline code has been adapted from [Efficient-3DCNNs GitHub repository](https://github.com/okankop/Efficient-3DCNNs), which is licensed under the MIT license. Additionally, parts of the fusion implementation are based on the [timm library](https://github.com/rwightman/pytorch-image-models), available under the Apache 2.0 license. For data preprocessing, we utilized [facenet-pytorch](https://github.com/timesler/facenet-pytorch).

## Migration: Changed CLI Flags

To better match the paper's specification, we have updated several CLI defaults and command-line arguments in `opts.py`:

- **Datasets**: We now support `--dataset RAVDESS`, `--dataset CMU_MOSEI`, and `--dataset CREMA_D` through a unified dataset registry.
- **Classes**: The `--n_classes` flag is now intelligently inferred if omitted (RAVDESS: 8, CMU_MOSEI: 6, CREMA_D: 6).
- **Attention Heads**: The old `--num_heads` flag has been replaced. Please use `--transformer_heads` and `--cross_attention_heads` to precisely configure the attention components.
- **Defaults updated to paper configuration**: 
  - `--batch_size` is now `8`
  - `--n_epochs` is now `128`
  - `--learning_rate` is now `0.01`
  - `--optimizer` is now `'adam'`
- **Testing defaults**: The codebase now defaults to `--test False`. Pass `--test` directly to execute the test suite explicitly.

## Loading Checkpoints Safely

The repository distinguishes between safe generic model loading and restoring complete trainer states:
- **Resumed Training vs State Dicts**: By default, `--resume_path` will safely load models specifying `weights_only=True` via PyTorch, which is generally secure. 
- **Trusted Resumes**: To load completely trusted `results/model.pth` or arbitrary pickles which resume entire workflows (including optimizers), please append the `--unsafe_resume` flag. Doing this runs native `torch.load` globally.

## Reproduction Steps

Exact commands for reconstructing the models per the paper:

### RAVDESS
```bash
python main.py --dataset RAVDESS --batch_size 8 --n_epochs 128 --learning_rate 0.01 --optimizer adam --transformer_heads 4 --cross_attention_heads 1
```

### CMU-MOSEI
```bash
python main.py --dataset CMU_MOSEI --batch_size 8 --n_epochs 128 --learning_rate 0.01 --optimizer adam --transformer_heads 4 --cross_attention_heads 1
```

### CREMA-D
```bash
python main.py --dataset CREMA_D --batch_size 8 --n_epochs 128 --learning_rate 0.01 --optimizer adam --transformer_heads 4 --cross_attention_heads 1
```
