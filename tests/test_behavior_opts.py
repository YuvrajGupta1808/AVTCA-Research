"""Tests for the --behavior flag (opts) and model factory wiring.

opts.py and factory.py are loaded standalone by path so importing them does not
run src/__init__ (which pulls librosa, absent in the test venv).
"""

import importlib.util
import os
from types import SimpleNamespace

_HERE = os.path.dirname(__file__)


def _load(name, relpath):
    path = os.path.normpath(os.path.join(_HERE, "..", relpath))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


opts = _load("opts_standalone", "src/config/opts.py")
factory = _load("factory_standalone", "src/models/factory.py")


def _opt(**overrides):
    base = dict(
        model="multimodal_cnn", n_classes=4, fusion="it", sample_duration=15,
        num_heads=1, pretrain_path="None", device="cpu",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_behavior_flag_defaults_off():
    o = opts.parse_opts([])
    assert o.behavior is False
    assert o.behavior_feature_dim == 22
    assert o.behavior_skip_dim == 64


def test_behavior_flag_on():
    o = opts.parse_opts(["--behavior"])
    assert o.behavior is True


def test_behavior_paths_default_empty():
    o = opts.parse_opts([])
    assert o.behavior_dir == ""
    assert o.behavior_baselines == ""


def test_factory_builds_behavior_model():
    model, _ = factory.generate_model(_opt(behavior=True))
    assert model.behavior is True
    assert hasattr(model, "classifier_fused")
    assert model.classifier_fused.in_features == 256 + 128 + 64


def test_factory_off_has_no_behavior():
    model, _ = factory.generate_model(_opt())
    assert model.behavior is False
    assert not hasattr(model, "classifier_fused")


def test_factory_builds_text_fusion_model():
    model, _ = factory.generate_model(_opt(text_fusion=True))
    assert model.text_fusion is True
    assert hasattr(model, "classifier_fused")
    assert model.classifier_fused.in_features == 256 + 128
    # text fusion disables the hashed late-text add-on
    assert model.late_text_fusion is False


def test_text_fusion_flag_parses():
    assert opts.parse_opts([]).text_fusion is False
    assert opts.parse_opts(["--text_fusion"]).text_fusion is True
    assert opts.parse_opts([]).text_backend == "hashing"
