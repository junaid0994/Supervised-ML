# Supervised Machine Learning Notebooks

A collection of Jupyter notebooks where I practice core supervised learning algorithms with scikit-learn. They cover both **regression** (predicting a number) and **classification** (predicting a category), along with feature engineering, regularization, and hyperparameter tuning.

## Contents

- [Repository structure](#repository-structure)
- [Notebooks](#notebooks)
- [Key takeaways](#key-takeaways)
- [Concepts practiced](#concepts-practiced)
- [Setup](#setup)
- [Tech stack](#tech-stack)
- [Next steps](#next-steps)

## Repository structure

```
Supervised-ML/
├── Linear_Regression.ipynb
├── Feature_Engineering.ipynb
├── Lasso_Regression.ipynb
├── Logistic_regression.ipynb
├── Naive_Bayes.ipynb
├── KNN.ipynb
└── README.md
```

## Notebooks

### Regression: Medical Insurance Charges (`insurance.csv`)

Goal: predict the insurance `charges` of a person from `age`, `sex`, `bmi`, `children`, `smoker` and `region`.

| Notebook | What it covers |
| --- | --- |
| `Linear_Regression.ipynb` | Baseline linear regression. Scatter plot of BMI vs charges coloured by smoker status, label encoding of `sex` and `smoker`, train/test split, R² and adjusted R². |
| `Feature_Engineering.ipynb` | Improves the baseline: one-hot encoding of `region` (`drop_first=True` to avoid the dummy variable trap) and interaction features `age_smoker` and `bmi_smoker`. Compares train vs test R² to check for overfitting or underfitting. |
| `Lasso_Regression.ipynb` | Lasso (L1) regularization. Tests different `alpha` values, plots MSE against alpha, then picks the best alpha automatically using `LassoCV` with 5-fold cross validation. |

### Classification: Heart Disease (`heart.csv`)

Goal: predict whether a patient has heart disease (`target` column).

| Notebook | What it covers |
| --- | --- |
| `Logistic_regression.ipynb` | Logistic regression, evaluated with accuracy, precision, recall and F1. Also shows the effect of `StandardScaler`. |
| `Naive_Bayes.ipynb` | Gaussian Naive Bayes with the same evaluation metrics. |
| `KNN.ipynb` | K-Nearest Neighbors with feature scaling. Compares k = 3, 5, 7, then tunes `n_neighbors` with `GridSearchCV` (5-fold CV, scored on recall). Finishes with a `Pipeline` (scaler + KNN) so scaling happens inside each CV fold and avoids data leakage. |

**KNN results (80/20 split, `random_state=42`):**

| k | Accuracy | Precision | Recall |
| --- | --- | --- | --- |
| 3 | 85.2% | 92.6% | 78.1% |
| 5 | 90.2% | 93.3% | 87.5% |
| 7 | **91.8%** | 93.5% | **90.6%** |

`GridSearchCV` over k = 3, 5, 7, 9 also selected **k = 7**.

## Key takeaways

- Interaction features such as `age_smoker` and `bmi_smoker` can improve a linear model without changing the algorithm.
- Comparing train and test scores is a quick way to spot overfitting.
- Lasso regularization with `LassoCV` removes the guesswork when choosing `alpha`.
- Distance-based models like KNN need feature scaling.
- For medical data, recall matters, so KNN was tuned on recall instead of accuracy.
- Putting the scaler inside a `Pipeline` keeps test data out of the scaling step during cross validation.

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

**1. Install the requirements**

```
pip install pandas seaborn matplotlib scikit-learn jupyter
```

**2. Add the datasets**

The data files are not included in this repository (they are listed in `.gitignore`). Download them and place them in the same folder as the notebooks:

- `insurance.csv`: the "Medical Cost Personal Dataset" (available on Kaggle)
- `heart.csv`: the Heart Disease dataset (available on Kaggle / UCI)

The notebooks load them with `pd.read_csv("insurance.csv")` and `pd.read_csv("heart.csv")`, so if you keep the files elsewhere, update those paths.

**3. Run**

```
jupyter notebook
```

Open any notebook and run the cells from top to bottom.

## Tech stack

Python, pandas, seaborn, matplotlib, scikit-learn, Jupyter Notebook

## Next steps

- Add Decision Tree and Random Forest notebooks
- Compare all classification models on the same metrics
