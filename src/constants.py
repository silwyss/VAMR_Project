from pathlib import Path

# Directories
ROOT_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = ROOT_DIR / 'Dataset'
IMAGES_DIR = DATASET_DIR / 'images'
CALIBRATION_DIR = ROOT_DIR / 'Calibration'
CALIBRATION_FILE = CALIBRATION_DIR / '00_calib_results' / 'camchain-mediajaviJAVISdatasetshwdscalibratione2kalibr.yaml'
OUTPUT_DIR = ROOT_DIR / 'outputs'
SFM_OUTPUT_DIR = OUTPUT_DIR / 'SfM'

# Parameters to build Map using SfM
SFM_ROTATION_THRESHOLD_DEG = 3   # used to select frames from dataset (better than just every n-th frame)
SFM_TRANSLATION_THRESHOLD = 0.04 # used to select frames from dataset (better than just every n-th frame)
