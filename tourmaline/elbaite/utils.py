import os
import pickle


ASSET_PATH = "../model"

def load_asset(filename: str, asset_path: str=ASSET_PATH):
    print(f"Loading asset: {filename}")
    with open(os.path.join(asset_path, filename), "rb") as f:
        asset = pickle.load(f)
    return asset