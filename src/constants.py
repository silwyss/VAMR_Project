from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = ROOT_DIR / 'Dataset'
IMAGES_DIR = DATASET_DIR / 'images'
CALIBRATION_DIR = ROOT_DIR / 'Calibration'
CALIBRATION_FILE = CALIBRATION_DIR / '00_calib_results' / 'camchain-mediajaviJAVISdatasetshwdscalibratione2kalibr.yaml'
OUTPUT_DIR = ROOT_DIR / 'outputs'
