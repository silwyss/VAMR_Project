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
METRICS_FILE = OUTPUT_DIR / 'metrics.json'
TABLES_FILE = OUTPUT_DIR / 'tables.md'

SFM_CONFIG_FILE = SFM_OUTPUT_DIR / 'map_config.json'       # parameters the database was built with
SFM_MAP_FRAMES_FILE = SFM_OUTPUT_DIR / 'map_frames.txt'    # keyframes used for the map
SFM_QUERY_FRAMES_FILE = SFM_OUTPUT_DIR / 'query_frames.txt' # held-out frames (queries for localization)

# Parameters to build Map using SfM -> build_map() rebuilds the database automatically if any of the
# frame selection / matching parameters change (they are stored in SFM_CONFIG_FILE)
SFM_ROTATION_THRESHOLD_DEG = 3   # used to select frames from dataset (better than just every n-th frame)
SFM_TRANSLATION_THRESHOLD_M = 0.04 # used to select frames from dataset (better than just every n-th frame)
SFM_MAP_IMAGE_COUNT_THRESHOLD = 20 # Minimum number of images in a map (from COLMAP incremental map) to count it as valid
SFM_MAP_POINT_COUNT_THRESHOLD = 300 # Minimum number of points in a map to count it as valid
SFM_MAP_MAX_ROTATION_ERROR_DEG = 5.0 # Max. median orientation error of an aligned model vs. the GT to count it as valid
SFM_MAP_MAX_POSITION_ERROR_M = 0.10  # Max. position error of an aligned model vs. the GT at the percentile below
SFM_MAP_POSITION_ERROR_PERCENTILE = 90 # Percentile used for the position check (catches partly broken models)
SFM_SEQUENTIAL_OVERLAP = 15 # number of following keyframes each keyframe is matched against

# Evaluation parameters
EVAL_RECALL_THRESHOLDS = ((0.02, 2.0), (0.05, 5.0), (0.10, 10.0)) # (metres, degrees) pose-error thresholds for recall
EVAL_FAST_MOTION_DEG_S = 90.0 # frames rotating faster than this count as "fast motion" in the slow/fast split
EVAL_RPE_DELTA = 10           # keyframe distance for the relative pose error (drift)
EVAL_MAP_CROP_RADIUS_M = 0.7 # horizontal radius around the scene centre shown in the zoomed map figure
