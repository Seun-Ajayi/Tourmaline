import os
import pandas as pd

from elbaite.ml.data import process_data
from elbaite.ml.model import compute_model_metrics
from elbaite.utils import load_asset
from elbaite.train import cat_features


def evaluate_model(
    data, 
    cat_cols,
    output_dir,
    model=None,
    encoder=None,
    lb=None
):
    model = model or load_asset("trained_model.pkl")
    encoder = encoder or load_asset("encoder.pkl")
    lb = lb or load_asset("lb.pkl")

    performance_df = pd.DataFrame(columns=["feature", "precision", "recall", "fbeta"])
    for feature in cat_cols:
        feature_performance = []
        for category in data[feature].unique():
            mask = data[feature] == category
            subset = data[mask]
            X_test, y_test, *_ = process_data(
                subset,
                categorical_features=cat_cols,
                encoder=encoder,
                label="salary",
                training=False,
                lb=lb
            )
            y_pred = model.predict(X_test)
            precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

            feature_performance.append(
                {
                    "feature": feature,
                    "category": category,
                    "precision": precision,
                    "recall": recall,
                    "fbeta": fbeta,
                }
            )

        performance_df = performance_df.append(feature_performance, ignore_index=True)
    output_file = os.path.join(output_dir, "slice_output.txt")
    performance_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    data = pd.read_csv("../data/clean_census.csv")
    output_dir = "../outputs"
    os.makedirs(output_dir, exist_ok=True)
    evaluate_model(data, cat_features, output_dir)
        


