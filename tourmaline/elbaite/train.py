""" Script to train machine learning model. """

import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
from evaluate import evaluate

OUTPUT = "../outputs"
os.makedirs(OUTPUT, exist_ok=True)
MODEL = "../model"
os.makedirs(MODEL, exist_ok=True)

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

def main(cat_cols: list=cat_features, datapath: str="../data/clean_census.csv"):

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
    assets_path = MODEL
    assets = [model, encoder, lb]
    assets_filenames = ["trained_model.pkl", "encoder.pkl", "lb.pkl"]

    for name, asset in zip(assets_filenames, assets):
        with open(os.path.join(assets_path, name), "wb") as f:
            pickle.dump(asset, f)

    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    evaluate(data, cat_cols, OUTPUT, model, encoder, lb)
    return model, precision, recall, fbeta

if __name__ == "__main__":
    _, precision, recall, fbeta = main()
    print(f"Precision: {precision}")
    print(f"recall: {recall}")
    print(f"fbeta: {fbeta}")


