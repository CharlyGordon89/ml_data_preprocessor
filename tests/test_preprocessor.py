from ml_preprocessor.preprocessor import DataPreprocessor
import pandas as pd

def test_preprocessor_transform():
    df = pd.DataFrame({
        "age": [25, 30, None],
        "income": [50000, 60000, 70000],
        "gender": ["M", "F", "F"]
    })

    preprocessor = DataPreprocessor(num_cols=["age", "income"], cat_cols=["gender"])
    transformed = preprocessor.fit_transform(df)
    assert transformed.shape[0] == 3
