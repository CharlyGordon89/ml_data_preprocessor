# ml_preprocessor/preprocessor.py

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, num_cols=None, cat_cols=None):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.pipeline = None

    def fit(self, X, y=None):
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        self.pipeline = ColumnTransformer(transformers=[
            ("num", numeric_pipeline, self.num_cols),
            ("cat", categorical_pipeline, self.cat_cols)
        ])

        self.pipeline.fit(X)
        return self

    def transform(self, X):
        return self.pipeline.transform(X)
