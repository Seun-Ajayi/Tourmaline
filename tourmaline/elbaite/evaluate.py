import os
import pandas as pd

from ml.data import process_data
from ml.model import compute_model_metrics

def evaluate(
    data, 
    cat_cols,
    output_dir,
    model=None,
    encoder=None,
    lb=None
):
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
                label="'salary",
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
                    "fbeta": fbeta
                }
            )

        performance_df.append(feature_performance, ignore_index=True)
    output_file = os.path.join(output_dir, "slice_output.txt")
    performance_df.to_csv(output_file, index=False)

        


