"""Core training and evaluation package for AVTCA research experiments."""

__all__ = ['main']


def __getattr__(name):
    # Imported lazily so that importing any src.* module does not pull in the whole
    # training stack (datasets, librosa, the trainer). `from src import main` still works.
    if name == 'main':
        from src.cli.train import main
        return main
    raise AttributeError(f"module 'src' has no attribute {name!r}")
