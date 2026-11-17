from pathlib import Path

# Directories
ROOT_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = ROOT_DIR / 'Dataset'
IMAGES_DIR = DATASET_DIR / 'images'
CALIBRATION_DIR = ROOT_DIR / 'Calibration'
CALIBRATION_FILE = CALIBRATION_DIR / '00_calib_results' / 'camchain-mediajaviJAVISdatasetshwdscalibratione2kalibr.yaml'
OUTPUT_DIR = ROOT_DIR / 'outputs'
FIGURES_DIR = OUTPUT_DIR / 'figures'
SFM_OUTPUT_DIR = OUTPUT_DIR / 'SfM'
SFM_GT_MAP = SFM_OUTPUT_DIR / 'known_poses'
SFM_MAP = SFM_OUTPUT_DIR / 'aligned'

# Parameters to build Map using SfM
SFM_ROTATION_THRESHOLD_DEG = 3   # used to select frames from dataset (better than just every n-th frame)
SFM_TRANSLATION_THRESHOLD = 0.04 # used to select frames from dataset (better than just every n-th frame)
SFM_MAP_IMAGE_COUNT_THRESHOLD = 20 # Minimum number of images in a map (from COLMAP incremental map) to count it as valid
SFM_MAP_POINT_COUNT_THRESHOLD = 300 # Minimum number of points in a map to count it as valid