# Supervised Machine Learning Notebooks

Jupyter notebooks where I practice core supervised learning algorithms with scikit-learn. They cover **regression** (predicting a number) and **classification** (predicting a category), plus feature engineering, regularization and hyperparameter tuning.

## Repository structure

```
Supervised-ML/
├── Linear_Regression.ipynb
├── Feature_Engineering.ipynb
├── Lasso_Regression.ipynb
├── Logistic_regression.ipynb
├── Naive_Bayes.ipynb
├── KNN.ipynb
├── SVM.ipynb
├── SVM(Regressor).ipynb
├── Decision_Tree.ipynb
└── README.md
```

Datasets are **not** included in this repo. See [Setup](#setup).

## Notebooks

### Regression: medical insurance charges (`insurance.csv`)

Goal: predict `charges` from `age`, `sex`, `bmi`, `children`, `smoker` and `region`.

Suggested order: Linear Regression → Feature Engineering → Lasso Regression.

| Notebook | What it covers |
| --- | --- |
| `Linear_Regression.ipynb` | Baseline linear regression. Scatter plot of BMI vs charges coloured by smoker status, label encoding of `sex` and `smoker`, train/test split, R² and adjusted R². |
| `Feature_Engineering.ipynb` | One-hot encoding of `region` (`drop_first=True` to avoid the dummy variable trap) and interaction features `age_smoker` and `bmi_smoker`. Compares train vs test R² to check for overfitting or underfitting. |
| `Lasso_Regression.ipynb` | Lasso (L1) regularization. Tests different `alpha` values, plots MSE against alpha, then picks the best alpha with `LassoCV` (5-fold cross validation). |

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

### Support Vector Machines and Decision Tree

<!-- TODO: add dataset, results and what each notebook covers -->

| Notebook | What it covers |
| --- | --- |
| `SVM.ipynb` | Support Vector Machine (SVC) classifier. |
| `SVM(Regressor).ipynb` | Support Vector Regression (SVR). |
| `Decision_Tree.ipynb` | Decision Tree model. |

## Concepts practiced

- Train/test split and reproducibility with `random_state`
- Encoding categorical variables (label mapping, one-hot encoding)
- Feature engineering with interaction terms
- Evaluation metrics: R², adjusted R², MSE, accuracy, precision, recall, F1
- Detecting overfitting by comparing train and test scores
- Regularization (Lasso) and choosing `alpha` with cross validation
- Feature scaling with `StandardScaler`
- Hyperparameter tuning with `GridSearchCV`
- Building `Pipeline`s to prevent data leakage

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

**3. Add the datasets**

Download these and place them in the same folder as the notebooks:

- `insurance.csv`: "Medical Cost Personal Dataset" (Kaggle)
- `heart.csv`: Heart Disease dataset (Kaggle / UCI)

The notebooks load them with `pd.read_csv("insurance.csv")` and `pd.read_csv("heart.csv")`. If you keep the files elsewhere, update those paths.

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
