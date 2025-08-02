# ml_preprocessor

`ml_preprocessor` is a modular, reusable Python package that provides standardized preprocessing for structured machine learning data.  
It is designed to be integrated into any ML pipeline with minimal effort while ensuring consistency, testability, and scalability.

---

## ✅ Features

- 🧼 Handles missing values (numeric + categorical)
- 🔢 Scales numerical features (StandardScaler)
- 🔤 Encodes categorical variables (OneHotEncoder)
- 🔄 Scikit-learn compatible: can be used inside `Pipeline` and `ColumnTransformer`
- ✅ Built-in unit tests
- 🧱 Clean structure ready for extension (e.g., log transforms, rare label grouping)
- ☁️ Cloud-compatible: can be plugged into AWS/GCP ML workflows

---

## 📦 Installation

### Local development (editable mode):

```bash
git clone https://github.com/CharlyGordon89/ml_preprocessor.git
cd ml_preprocessor
pip install -e .
