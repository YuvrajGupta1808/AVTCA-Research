"""--behavior_frames: the OpenFace series must follow the video clock by default."""

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


opts = _load("opts_standalone_bf", "src/config/opts.py")


def _resolve():
    # src/data/dataset.py imports the dataset registry (librosa etc.); re-implement
    # the resolver contract check against the function source to stay import-light.
    src = open(os.path.normpath(os.path.join(_HERE, "..", "src", "data", "dataset.py"))).read()
    ns = {}
    start = src.index("def resolve_behavior_frames")
    exec(src[start:], ns)
    return ns["resolve_behavior_frames"]


def test_flag_default_zero():
    assert opts.parse_opts([]).behavior_frames == 0
    assert opts.parse_opts(["--behavior_frames", "50"]).behavior_frames == 50


def test_default_follows_video_cap():
    resolve = _resolve()
    assert resolve(SimpleNamespace(behavior_frames=0, max_video_frames=50)) == 50
    assert resolve(SimpleNamespace(behavior_frames=0, max_video_frames=96)) == 96


def test_explicit_wins_and_zero_cap_falls_back():
    resolve = _resolve()
    assert resolve(SimpleNamespace(behavior_frames=15, max_video_frames=50)) == 15
    assert resolve(SimpleNamespace(behavior_frames=0, max_video_frames=0)) == 15
    assert resolve(SimpleNamespace()) == 15
