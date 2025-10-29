# 🏠 House Price Prediction using Linear Regression

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](#license)
[![Model: Linear Regression](https://img.shields.io/badge/model-Linear%20Regression-orange.svg)](#modeling--evaluation)
[![Dataset rows](https://img.shields.io/badge/rows-545-lightgrey.svg)](#dataset)
[![Status](https://img.shields.io/badge/status-Completed-green.svg)](#summary)

One-line description
A Multiple Linear Regression project to predict house prices from structural and locational features using basic preprocessing, feature encoding, scaling, and model evaluation.

Table of Contents
- [Summary](#summary)
- [Dataset](#dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Preprocessing](#preprocessing)
- [Feature Engineering & Selection](#feature-engineering--selection)
- [Modeling & Evaluation](#modeling--evaluation)
- [Reproduce / Quick Start](#reproduce--quick-start)
- [Project Structure (suggested)](#project-structure-suggested)
- [Results](#results)
- [Limitations & Next Steps](#limitations--next-steps)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

Summary
-------
This project trains a Multiple Linear Regression model to estimate house prices using a dataset of 545 examples and 13 columns. The pipeline includes data cleaning checks (no missing values), categorical encoding, scaling of continuous features, correlation-based EDA, an 80/20 train/test split, and model evaluation. The model achieves an R-squared score of ~0.649 on the test set.

Dataset
-------
- Size: 545 rows, 13 columns.
- Typical columns (example):
  - price (target)
  - area variables (e.g., area, lotsize)
  - binary categorical flags (mainroad, guestroom, basement, hotwaterheater, etc.)
  - furnishingstatus (furnished / semi-furnished / unfurnished)
  - other numeric and categorical features representing structure and location
- Missing data: none (checked and confirmed).

Exploratory Data Analysis (EDA)
-------------------------------
- Summary statistics (mean, median, std) were reviewed for all features.
- A correlation matrix (visualized as a heatmap) was inspected to identify strong predictors and multicollinearity candidates.
- Distribution plots and scatter plots were used to check skewness and relationships with price.

Preprocessing
-------------
- Binary categorical features: 'yes'/'no' converted to 1/0 (e.g., mainroad, guestroom, basement).
- furnishingstatus encoded to numeric ordinal values:
  - furnished -> 1
  - semi-furnished -> 2
  - unfurnished -> 3
- Continuous features (price and area-related features) scaled using sklearn.preprocessing.StandardScaler.
  - Note: Scaling the target (price) is optional depending on downstream needs; in this project price was scaled as described.
- Train/test split: 80% training, 20% testing (use a fixed random_state for reproducibility, e.g., 42).

Feature Engineering & Selection
-------------------------------
- Correlation heatmap was used to visualize relationships between features and the target to guide feature selection.
- Basic selection based on correlation magnitude and domain knowledge (remove or combine highly colinear features where appropriate).

Modeling & Evaluation
---------------------
- Algorithm: sklearn.linear_model.LinearRegression (ordinary least squares).
- Training: fit on training set (80%).
- Evaluation:
  - R-squared on test set: 0.649 (≈ 65% of variance explained).
  - Consider recording additional metrics such as MAE and RMSE for practical error interpretation.

Example training / evaluation pseudocode (sklearn)
--------------------------------------------------
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# load
df = pd.read_csv("data/house_prices.csv")

# encode binary columns
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheater']  # example
for c in binary_cols:
    df[c] = df[c].map({'yes': 1, 'no': 0})

# encode furnishingstatus
df['furnishingstatus'] = df['furnishingstatus'].map({
    'furnished': 1,
    'semi-furnished': 2,
    'unfurnished': 3
})

# split
X = df.drop(columns=['price'])
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# scale continuous features (example: area, price if desired)
scaler = StandardScaler()
cont_features = ['area']  # add continuous columns here
X_train[cont_features] = scaler.fit_transform(X_train[cont_features])
X_test[cont_features] = scaler.transform(X_test[cont_features])

# train
model = LinearRegression()
model.fit(X_train, y_train)

# eval
y_pred = model.predict(X_test)
print("R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
```

Reproduce / Quick Start
-----------------------
1. Clone the repo:
   git clone https://github.com/your-org/your-repo.git
2. Create a virtual environment and install dependencies:
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
3. Prepare data:
   - Place the CSV as data/house_prices.csv or update path in scripts/notebook.
4. Run training (example):
   python src/train.py
   or open notebooks/eda_and_model.ipynb and run cells.
5. Evaluate:
   python src/evaluate.py

Project Structure (suggested)
-----------------------------
- data/
  - house_prices.csv
- notebooks/
  - eda_and_model.ipynb
- src/
  - data_preprocessing.py
  - features.py
  - train.py
  - evaluate.py
  - predict.py
- requirements.txt
- README.md
- LICENSE

Results
-------
- Test R-squared: 0.649 (model explains ≈65% of variance in house prices).
- Additional metrics (recommended to log):
  - MAE: (report if computed)
  - RMSE: (report if computed)
- Interpretation: Linear Regression captures many linear relationships in the dataset, but ~35% of variance remains unexplained, suggesting non-linearity, omitted variables, or noise.

Limitations & Next Steps
------------------------
- Limitations:
  - Linear model assumes linear relationships and may miss nonlinear patterns.
  - Encoding furnishingstatus as ordinal may or may not reflect true ordering — consider one-hot if orderless.
  - Scaling the target (price) can affect interpretability of coefficients.
  - Possible multicollinearity between area-related features.
- Next steps / improvements:
  - Apply regularization (Ridge, Lasso) to reduce overfitting and produce more stable coefficients.
  - Use K-Fold cross-validation for more robust evaluation.
  - Try nonlinear models (Random Forest, Gradient Boosting, XGBoost) and compare performance.
  - Hyperparameter tuning (GridSearchCV / RandomizedSearchCV).
  - Add interaction features or polynomial features where appropriate.
  - Conduct residual analysis, outlier detection, and leverage domain-specific features (neighborhood data, year built, proximity to amenities).
  - Deploy a simple API (Flask/FastAPI) for predictions and a minimal UI for demonstration.

Contributing
------------
- Fork the repository and create a feature branch: git checkout -b feat/your-feature
- Run tests and linters before opening a PR.
- Add a clear description and reproducible steps in PRs.

License
-------
Open source

Contact
-------
Maintainer: Your Name — thanusivanallaperumal.com


Acknowledgements
----------------
Thanks to the dataset authors and the open-source libraries used in this project (pandas, scikit-learn, matplotlib / seaborn).
