# Supervised Machine Learning Notebooks

Jupyter notebooks where I practice core supervised learning algorithms with scikit-learn. They cover **regression** (predicting a number) and **classification** (predicting a category), plus feature engineering, regularization, hyperparameter tuning and ensemble methods.

## Repository structure

```
Supervised-ML/
├── Linear_Regression.ipynb
├── Feature_Engineering.ipynb
├── Lasso_Regression.ipynb
├── SVM(Regressor).ipynb
├── Logistic_regression.ipynb
├── Naive_Bayes.ipynb
├── KNN.ipynb
├── SVM.ipynb
├── Decision_Tree.ipynb
├── Random_Forest.ipynb
└── README.md
```

Some datasets are **not** included in this repo. See [Setup](#setup).

## Notebooks

### Regression: medical insurance charges (`insurance.csv`)

Goal: predict `charges` from `age`, `sex`, `bmi`, `children`, `smoker` and `region`.

Suggested order: Linear Regression → Feature Engineering → Lasso Regression.

| Notebook | What it covers |
| --- | --- |
| `Linear_Regression.ipynb` | Baseline linear regression. Scatter plot of BMI vs charges coloured by smoker status, label encoding of `sex` and `smoker`, train/test split, R² and adjusted R². |
| `Feature_Engineering.ipynb` | One-hot encoding of `region` (`drop_first=True` to avoid the dummy variable trap) and interaction features `age_smoker` and `bmi_smoker`. Compares train vs test R² to check for overfitting or underfitting. |
| `Lasso_Regression.ipynb` | Lasso (L1) regularization. Tests different `alpha` values, plots MSE against alpha, then picks the best alpha with `LassoCV` (5-fold cross validation). |

### Regression: diabetes progression (scikit-learn `load_diabetes`)

Goal: predict a numeric measure of diabetes progression from 10 baseline features.

| Notebook | What it covers |
| --- | --- |
| `SVM(Regressor).ipynb` | Support Vector Regression (`SVR`). The target is standardized with `StandardScaler`. Compares the default RBF, linear and polynomial kernels, then searches `C`, `kernel` and `epsilon` with `GridSearchCV` (5-fold CV, scored on R²). |

**SVR results** (70/30 split, `random_state=42`, R² on the scaled target):

| Model | Train R² | Test R² |
| --- | --- | --- |
| Default (RBF kernel) | 0.660 | 0.488 |
| Linear kernel | 0.452 | 0.443 |
| Polynomial kernel | 0.579 | 0.242 |
| Linear, `C=10`, `epsilon=0.1` (after `GridSearchCV`) | 0.515 | 0.474 |

The polynomial kernel overfits the most (large gap between train and test R²). `GridSearchCV` selected the linear kernel.

### Classification: heart disease (`heart.csv`)

Goal: predict whether a patient has heart disease (`target` column).

| Notebook | What it covers |
| --- | --- |
| `Logistic_regression.ipynb` | Logistic regression, evaluated with accuracy, precision, recall and F1. Shows the effect of `StandardScaler`. |
| `Naive_Bayes.ipynb` | Gaussian Naive Bayes with the same evaluation metrics. |
| `KNN.ipynb` | K-Nearest Neighbors with feature scaling. Compares k = 3, 5, 7, then tunes `n_neighbors` with `GridSearchCV` (5-fold CV, scored on recall). Ends with a `Pipeline` (scaler + KNN) so scaling happens inside each CV fold and avoids data leakage. |

**KNN results** (80/20 split, `random_state=42`):

| k | Accuracy | Precision | Recall |
| --- | --- | --- | --- |
| 3 | 85.2% | 92.6% | 78.1% |
| 5 | 90.2% | 93.3% | 87.5% |
| 7 | **91.8%** | 93.5% | **90.6%** |

`GridSearchCV` over k = 3, 5, 7, 9 also selected **k = 7**.

### Classification: iris flowers (scikit-learn `load_iris`)

Goal: predict the iris species (3 classes) from sepal and petal measurements.

| Notebook | What it covers |
| --- | --- |
| `SVM.ipynb` | Support Vector Classifier (`SVC`) with a linear kernel and `C=1`, on standardized features. |

**Result** (70/30 split, `random_state=42`): accuracy **97.8%**.

### Classification: Titanic survival (seaborn `titanic` dataset)

Goal: predict whether a passenger survived (`survived`) from `pclass`, `age`, `embarked`, `sex` and `fare`.

Preprocessing in both notebooks: median imputation for `age`, most-frequent imputation for `embarked`, label encoding for `sex` and `embarked`, 80/20 split with `random_state=42`.

| Notebook | What it covers |
| --- | --- |
| `Decision_Tree.ipynb` | Decision tree without pruning, evaluated with accuracy, precision, recall and F1, plus a tree plot with `plot_tree`. |
| `Random_Forest.ipynb` | Decision tree with `max_depth=4`, then Random Forest (501 trees, OOB score), then Bagging with a decision tree and with logistic regression as base models. |

**Results** (test set):

| Model | Accuracy | Notes |
| --- | --- | --- |
| Decision tree, no pruning | 77.1% | Precision 70.9%, recall 75.7%, F1 73.2% |
| Decision tree, `max_depth=4` | 79.9% | Train accuracy 84.0% |
| Random Forest (501 trees, `max_depth=4`) | 81.0% | OOB score 81.7% |
| Bagging, decision tree base (201 estimators) | 79.9% | |
| Bagging, logistic regression base (201 estimators) | 79.9% | |

The Random Forest and Bagging models do not set `random_state`, so their scores can change slightly between runs.

## Concepts practiced

- Train/test split and reproducibility with `random_state`
- Encoding categorical variables (label mapping, one-hot encoding)
- Handling missing values with `SimpleImputer`
- Feature engineering with interaction terms
- Evaluation metrics: R², adjusted R², MSE, accuracy, precision, recall, F1
- Detecting overfitting by comparing train and test scores
- Regularization (Lasso) and choosing `alpha` with cross validation
- Feature scaling with `StandardScaler`
- Hyperparameter tuning with `GridSearchCV`
- Building `Pipeline`s to prevent data leakage
- SVM kernels (linear, polynomial, RBF) for classification and regression
- Controlling tree depth to reduce overfitting
- Ensemble methods: Random Forest, Bagging and the out-of-bag (OOB) score

## Setup

**1. Clone the repo**

```
git clone https://github.com/junaid0994/Supervised-ML.git
cd Supervised-ML
```

**2. Install the requirements**

```
pip install pandas seaborn matplotlib scikit-learn jupyter
```

**3. Get the datasets**

Download these and place them in the same folder as the notebooks:

- `insurance.csv`: "Medical Cost Personal Dataset" (Kaggle)
- `heart.csv`: Heart Disease dataset (Kaggle / UCI)

The notebooks load them with `pd.read_csv("insurance.csv")` and `pd.read_csv("heart.csv")`. If you keep the files elsewhere, update those paths.

No download is needed for the rest:

- Iris and Diabetes load from scikit-learn (`datasets.load_iris`, `datasets.load_diabetes`).
- Titanic loads with `sns.load_dataset("titanic")`, which needs an internet connection the first time.

**4. Run**

```
jupyter notebook
```

Open any notebook and run the cells from top to bottom.

## Tech stack

Python, pandas, seaborn, matplotlib, scikit-learn, Jupyter Notebook

## Author

Junaid Ashraf

- GitHub: [junaid0994](https://github.com/junaid0994)
- LinkedIn: [junaid-ashraf](https://www.linkedin.com/in/junaid-ashraf-53123b37a/)
