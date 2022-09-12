""" Script to train machine learning model. """

import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from tourmaline.elbaite.ml.data import process_data
from tourmaline.elbaite.ml.model import train_model, compute_model_metrics, inference
from tourmaline import PROJECT_DIR

OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_DIR = os.path.join(PROJECT_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)
DATA_PATH = os.path.join(PROJECT_DIR, "data/clean_census.csv")
cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

from tourmaline.elbaite.evaluate import evaluate_model

def main(cat_cols: list=cat_features, datapath: str=DATA_PATH):

    data = pd.read_csv(datapath)

    train, test = train_test_split(data, test_size=0.20)

    
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, *_ = process_data(
        test, 
        categorical_features=cat_features, 
        label="salary", 
        training=False, 
        encoder=encoder, 
        lb=lb
    )

    # Train and save a model.
    model = train_model(X_train, y_train)
    assets_path = MODEL_DIR
    assets = [model, encoder, lb]
    assets_filenames = ["trained_model.pkl", "encoder.pkl", "lb.pkl"]

    for name, asset in zip(assets_filenames, assets):
        with open(os.path.join(assets_path, name), "wb") as f:
            pickle.dump(asset, f)

    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    evaluate_model(data, cat_cols, OUTPUT_DIR, model, encoder, lb)
    return model, precision, recall, fbeta

if __name__ == "__main__":
    _, precision, recall, fbeta = main()
    print(f"Precision: {precision}")
    print(f"recall: {recall}")
    print(f"fbeta: {fbeta}")
