import os
import shutil

from src.constants import SFM_GT_MAP, SFM_MAP, SFM_OUTPUT_DIR
from src.SfM.build_map import build_map

def main():
    if not os.path.isdir(SFM_GT_MAP) or not os.path.isdir(SFM_MAP):
        # build the map if it does not exist yet
        # Full Map rebuild can also be forced by deleting outputs/SfM/database.db and calling build_map
        shutil.rmtree(SFM_OUTPUT_DIR, True)
        build_map()

if __name__ == '__main__':
    main()
