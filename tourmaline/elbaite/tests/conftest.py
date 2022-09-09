import pandas as pd
import pytest


@pytest.fixture()
def X_train():
    return pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

@pytest.fixture()
def y_train():
    return pd.Series([10, 11, 12])

@pytest.fixture()
def X_test():
    return X_train

@pytest.fixture()
def y_test():
    return y_train

@pytest.fixture()
def preds():
    return y_test

@pytest.fixture()
def test_model(preds):
    class ModelMocker:
        def __init__(self, preds=preds):
            self.preds = preds

        def predict(self, X):
            return self.preds

    return ModelMocker()


