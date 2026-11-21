# VAMR_Project

## Setup

Requires **Python 3.12+** (the code uses 3.12 f-string syntax).

```sh
python3.12 -m venv .venv
# activate: .venv\Scripts\Activate.ps1 (Windows)  |  source .venv/bin/activate (Linux/macOS)
python -m pip install --upgrade pip
pip install --no-deps -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

- `--no-deps` is needed: `requirements.txt` is a full freeze, and without it pip reports a false conflict between `hloc` and the pinned `lightglue` git commit.
- `torch` is pinned to a CUDA 13.0 build. Without a compatible NVIDIA GPU, install a matching `torch`/`torchvision` from [pytorch.org](https://pytorch.org/get-started/locally/) instead.

## Dataset

Download the EDS sequence `01_peanuts_light` as the **archive (tgz)** and extract it into `Dataset/`, then copy the image timestamps to where the loader expects them:

```sh
cp Dataset/images/timestamps.txt Dataset/times.txt
```

Expected layout: `Dataset/images/`, `Dataset/events.h5`, `Dataset/times.txt`, `Dataset/stamped_groundtruth.txt`.

## Running

Run files with F5 in VS Code (sets `PYTHONPATH` to the repo root so `src.*` imports work), or from the repo root with `python -m src.main`.
The map is built into `outputs/SfM/` (`known_poses/` = map triangulated from GT poses, used for localization).
