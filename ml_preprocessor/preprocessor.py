import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class StandardPreprocessor(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible preprocessor that handles missing values,
    numeric scaling, and categorical encoding.

    Attributes:
        strategy (str): Missing value imputation strategy ('mean', 'median', 'most_frequent')
        scale (bool): Whether to apply standard scaling to numeric columns
        encode (bool): Whether to one-hot encode categorical columns
    """

    def __init__(self, strategy="mean", scale=True, encode=True):
        """
        Initialize the preprocessor.

        Args:
            strategy (str): Imputation strategy ('mean', 'median', 'most_frequent')
            scale (bool): Whether to scale numeric features
            encode (bool): Whether to one-hot encode categorical features
        """
        self.strategy = strategy
        self.scale = scale
        self.encode = encode
        self.pipeline = None
        self.feature_names_ = None

    def fit(self, X, y=None):
        """
        Fit the preprocessing pipeline to the input data.

        Args:
            X (pd.DataFrame): Input features
            y: Ignored (compatibility)

        Returns:
            self
        """

        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        if X.empty:
            raise ValueError("Cannot fit preprocessor on an empty DataFrame.")

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        transformers = []

        if numeric_cols:
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy=self.strategy)),
                ('scaler', StandardScaler()) if self.scale else ('noop', 'passthrough')
            ])
            transformers.append(('num', num_pipeline, numeric_cols))

        if categorical_cols:
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="most_frequent")),
                ('encoder', OneHotEncoder(handle_unknown='ignore')) if self.encode else ('noop', 'passthrough')
            ])
            transformers.append(('cat', cat_pipeline, categorical_cols))

        self.pipeline = ColumnTransformer(transformers)
        self.pipeline.fit(X)

        # Save feature names if encoding is used
        self.feature_names_ = self._get_feature_names(numeric_cols, categorical_cols)

        return self

    def transform(self, X):
        """
        Apply the fitted transformations to new data.

        Args:
            X (pd.DataFrame): New input features

        Returns:
            np.ndarray or pd.DataFrame: Transformed feature matrix
        """
        if self.pipeline is None:
            raise RuntimeError("You must fit the preprocessor before calling transform().")

        transformed = self.pipeline.transform(X)

        try:
            return pd.DataFrame(transformed, columns=self.feature_names_, index=X.index)
        except Exception:
            return transformed  # fallback if feature names can't be resolved

    def _get_feature_names(self, numeric_cols, categorical_cols):
        """
        Generate output feature names after transformation (used for DataFrame output).

        Args:
            numeric_cols (list): List of numeric feature names
            categorical_cols (list): List of categorical feature names

        Returns:
            list: Output column names
        """
        output_features = []

        if self.scale:
            output_features += numeric_cols
        else:
            output_features += numeric_cols

        if self.encode:
            cat_encoder = self.pipeline.named_transformers_['cat'].named_steps.get('encoder', None)
            if cat_encoder:
                encoded_names = cat_encoder.get_feature_names_out(categorical_cols)
                output_features += encoded_names.tolist()
        else:
            output_features += categorical_cols

        return output_features
