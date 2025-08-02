# 🧠 ml_preprocessor

`ml_preprocessor` is a modular, production-ready Python package that provides standardized preprocessing for structured ML data.  
Designed for reusability, testability, and seamless integration with scikit-learn pipelines and real-world ML systems.

---

## ✅ Features

- 🧼 Robust handling of missing values (`mean`, `median`, `most_frequent`)
- 🔢 Numerical feature scaling (optional `StandardScaler`)
- 🔤 Categorical feature encoding (optional `OneHotEncoder`)
- 🔄 Fully scikit-learn compatible: use in `Pipeline`, `GridSearchCV`, or `ColumnTransformer`
- 📊 Returns clean `pandas.DataFrame` with column names (if possible)
- 🧪 Comprehensive unit test suite using `pytest`
- 🧱 Modular codebase designed for extension and configuration
- ☁️ Ready for MLOps: compatible with cloud pipelines and CI/CD testing

---

## 📦 Installation

### 💻 For local development:
```bash
git clone https://github.com/CharlyGordon89/ml_preprocessor.git
cd ml_preprocessor
pip install -e .
```

### 🧪 To run tests:
```bash
pytest -v tests/
```

---

## 🚀 Quick Start

```python
import pandas as pd
from ml_preprocessor import StandardPreprocessor

# Sample input
df = pd.DataFrame({
    "age": [25, 30, None, 22],
    "income": [50000, 60000, 55000, None],
    "gender": ["male", "female", "female", None],
    "country": ["US", "UK", "US", "UK"]
})

# Initialize preprocessor
pp = StandardPreprocessor(strategy="median", scale=True, encode=True)

# Fit and transform
X_processed = pp.fit_transform(df)

print(X_processed.head())
```

---

## 🧠 API: `StandardPreprocessor`

```python
StandardPreprocessor(
    strategy: str = "mean",   # missing value imputation: "mean", "median", "most_frequent"
    scale: bool = True,       # whether to apply StandardScaler to numeric features
    encode: bool = True       # whether to apply OneHotEncoder to categorical features
)
```

### Methods:
- `fit(X: pd.DataFrame)`: learns preprocessing steps
- `transform(X: pd.DataFrame)`: applies learned transformation
- `fit_transform(X)`: convenience method
- `feature_names_`: names of transformed features (if output is a DataFrame)

---

## 🔬 Testing Coverage

```bash
pytest -v
```

Covers:
- Imputation strategies
- Enabling/disabling scaling or encoding
- Input validation
- Edge cases (e.g., empty input, unseen categories)
- Output feature integrity

---

## 🛠️ Planned Enhancements

- [ ] Support for rare-label grouping
- [ ] Configurable column selection
- [ ] Integration with YAML configs (`ml-config`)
- [ ] Logging and pipeline tracing hooks
- [ ] Native support for TensorFlow / PyTorch tensors (optional)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -am 'Add feature'`)
4. Push to the branch (`git push origin feature-name`)
5. Open a pull request

---

## 📄 License

MIT License © 2025 CharlyGordon89