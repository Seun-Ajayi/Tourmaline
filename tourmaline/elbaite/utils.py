import os
import pickle
from tourmaline import PROJECT_DIR

PROJECT_FOLDER = PROJECT_DIR
ASSET_PATH = os.path.join(PROJECT_FOLDER, "model")

def load_asset(filename: str, asset_path: str=ASSET_PATH):
    print(f"Loading asset: {filename}")
    with open(os.path.join(asset_path,  filename), "rb") as f:
        asset = pickle.load(f)
    return asset