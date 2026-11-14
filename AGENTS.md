# Repository Guidelines

## Project Structure & Module Organization
Core training code lives in `src/`: `src/config/` holds CLI options, `src/data/` dataset and transform logic, `src/engine/` training/evaluation runtime, and `src/models/` model factories. Dataset-specific preprocessing scripts live under `preprocessing/` (`ravdess/`, `daisee/`, `engagenet/`, `mosei/`, `cremad/`). The Streamlit demo is in `ui/`. Tests live in `tests/`. Large runtime artifacts belong in `datasets/`, `pretrained/`, `results/`, `generated/`, and `logs/`; treat those as data/output directories, not source code.

## Build, Test, and Development Commands
Create and use a Conda environment for every session, for example: `conda create -n avtca python=3.10 && conda activate avtca`. Install the main environment with `pip install -r requirements.txt`. Start a training run with `python -m src.main --dataset RAVDESS --result_path results/dev_run`. Run a quick smoke test with `python -m src.main --dataset RAVDESS --n_epochs 2 --device cpu --no_val --result_path results/smoke`. Evaluate a checkpoint with `python src/evaluate.py --checkpoint results/my_run/...pth --result_path results/my_run --num_heads 8`. Launch the inference UI with `streamlit run ui/app.py`. Run tests with `python -m pytest tests/ -v`.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, `snake_case` for functions/files/CLI flags, `PascalCase` for classes, and concise module docstrings only when helpful. Keep new modules inside the current package layout rather than adding top-level scripts unless they are true utilities under `scripts/`. Prefer explicit CLI arguments over hard-coded paths. No formatter or linter is configured in the repo today, so match the surrounding file style closely and keep imports tidy.

## Testing Guidelines
Tests use `unittest` and are executed through `pytest`. Add new coverage in `tests/test_*.py`, mirroring the module or feature under change. Favor fast unit and smoke tests with synthetic tensors or temporary files over dataset-heavy integration runs. If you add a CLI option, dataset registry entry, or model path, include at least one regression test.

## Commit & Pull Request Guidelines
Recent commits use short, imperative summaries such as `Add DAISEE preprocessing and training support` and `Fix CI model factory option defaults`. Keep commit titles focused on one change. Pull requests should describe the affected dataset or pipeline, list commands run for verification, and call out any required assets, checkpoints, or annotation files. Include screenshots only for `ui/` changes.

## Data & Configuration Tips
Do not commit raw datasets, large checkpoints, or generated result folders unless the change explicitly requires tracked fixtures. Keep pretrained weights in `pretrained/` and write new experiment outputs to a dedicated subdirectory in `results/`.
