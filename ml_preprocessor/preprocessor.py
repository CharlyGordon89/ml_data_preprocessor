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


    def save(self, file_path: str = "artifacts/models/preprocessor.joblib"):
        """
        Save fitted preprocessor to disk.
        Path matches ml_project_template's artifacts/ structure.
        """
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, file_path)
        return file_path

    @classmethod
    def load(cls, file_path: str = "artifacts/models/preprocessor.joblib"):
        """
        Load saved preprocessor from disk.
        Returns a new DataPreprocessor instance with loaded pipeline.
        """
        preprocessor = cls(num_cols=[], cat_cols=[])  # Columns will be overwritten
        preprocessor.pipeline = joblib.load(file_path)
        return preprocessor


    def save_to_cloud(self, s3_path: str):
    import boto3
    buffer = io.BytesIO()
    joblib.dump(self.pipeline, buffer)
    s3 = boto3.client('s3')
    s3.put_object(Bucket=s3_path.bucket, Key=s3_path.key, Body=buffer.getvalue())
