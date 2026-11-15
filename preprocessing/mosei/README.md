# CMU-MOSEI preprocessing (unsupported)

These scripts are kept from an exploration of CMU-MOSEI (June 2026) and are **not supported**: the
training code has no MOSEI loader (there is no `datasets/mosei.py` and no `MOSEI` entry in
`src/data/dataset.py`), several defaults still point at the removed `mosei_preprocessing/` directory and at
`CMU-MOSEI/` instead of `datasets/CMU-MOSEI/`, and the scripts need extra packages and commands (`h5py`,
`imageio-ffmpeg`, `gdown`, `yt-dlp`). Nothing in the EngageNet paper uses them. Pass every path explicitly
if you run one, and expect to write the loader yourself.
