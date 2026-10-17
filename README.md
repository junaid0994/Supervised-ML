# Supervised Machine Learning with scikit-learn

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626)

Hands-on Jupyter notebooks that walk through core **supervised learning** algorithms, from a simple baseline to feature engineering, regularization and hyperparameter tuning. The repo covers both problem types:

- **Regression**: predicting a number (medical insurance charges)
- **Classification**: predicting a category (presence of heart disease)

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Projects](#projects)
  - [1. Regression: Medical Insurance Charges](#1-regression-medical-insurance-charges)
  - [2. Classification: Heart Disease](#2-classification-heart-disease)
- [Key Results](#key-results)
- [Concepts Practiced](#concepts-practiced)
- [Getting Started](#getting-started)
- [Tech Stack](#tech-stack)
- [Author](#author)

---

## Repository Structure

```
Supervised-ML/
├── Linear_Regression.ipynb       # Baseline regression
├── Feature_Engineering.ipynb     # One-hot encoding + interaction features
├── Lasso_Regression.ipynb        # L1 regularization with LassoCV
├── Logistic_regression.ipynb     # Classification baseline
├── Naive_Bayes.ipynb             # Gaussian Naive Bayes
├── KNN.ipynb                     # KNN, GridSearchCV and Pipeline
└── README.md
```

> The datasets are **not** included in the repo (they are listed in `.gitignore`). See [Getting Started](#getting-started).

---

## Projects

### 1. Regression: Medical Insurance Charges

**Dataset:** `insurance.csv` (Medical Cost Personal Dataset)
**Goal:** Predict a person's insurance `charges` from `age`, `sex`, `bmi`, `children`, `smoker` and `region`.

| Notebook | What it covers |
| --- | --- |
| [`Linear_Regression.ipynb`](Linear_Regression.ipynb) | Baseline linear regression. Scatter plot of BMI vs charges coloured by smoker status, label encoding of `sex` and `smoker`, train/test split, R² and adjusted R². |
| [`Feature_Engineering.ipynb`](Feature_Engineering.ipynb) | Improves the baseline with one-hot encoding of `region` (`drop_first=True` to avoid the dummy variable trap) and interaction features `age_smoker` and `bmi_smoker`. Compares train vs test R² to check for overfitting or underfitting. |
| [`Lasso_Regression.ipynb`](Lasso_Regression.ipynb) | Lasso (L1) regularization. Tests different `alpha` values, plots MSE against alpha, then selects the best alpha automatically with `LassoCV` and 5-fold cross validation. |

### 2. Classification: Heart Disease

**Dataset:** `heart.csv` (Heart Disease dataset)
**Goal:** Predict whether a patient has heart disease (the `target` column).

| Notebook | What it covers |
| --- | --- |
| [`Logistic_regression.ipynb`](Logistic_regression.ipynb) | Logistic regression evaluated with accuracy, precision, recall and F1. Also shows the effect of `StandardScaler`. |
| [`Naive_Bayes.ipynb`](Naive_Bayes.ipynb) | Gaussian Naive Bayes with the same evaluation metrics. |
| [`KNN.ipynb`](KNN.ipynb) | K-Nearest Neighbors with feature scaling. Compares k = 3, 5, 7, then tunes `n_neighbors` with `GridSearchCV` (5-fold CV, scored on recall). Finishes with a `Pipeline` (scaler + KNN) so scaling happens inside each CV fold, which avoids data leakage. |

---

## Key Results

**KNN on the heart disease data** (80/20 split, `random_state=42`):

| k | Accuracy | Precision | Recall |
| :-: | :-: | :-: | :-: |
| 3 | 85.2% | 92.6% | 78.1% |
| 5 | 90.2% | 93.3% | 87.5% |
| 7 | **91.8%** | 93.5% | **90.6%** |

`GridSearchCV` over k = 3, 5, 7, 9 also selected **k = 7**. Recall is the metric of focus here because missing a patient who has heart disease is costlier than a false alarm.

---

## Concepts Practiced

- Train/test split and reproducibility with `random_state`
- Encoding categorical variables (label mapping, one-hot encoding)
- Feature engineering with interaction terms
- Evaluation metrics: R², adjusted R², MSE, accuracy, precision, recall, F1
- Detecting overfitting by comparing train and test scores
- Regularization (Lasso) and choosing `alpha` with cross validation
- Feature scaling with `StandardScaler`
- Hyperparameter tuning with `GridSearchCV`
- Building `Pipeline`s to prevent data leakage

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/junaid0994/Supervised-ML.git
cd Supervised-ML
```

### 2. Install the requirements

```bash
pip install pandas seaborn matplotlib scikit-learn jupyter
```

### 3. Add the datasets

Download the following files and place them in the same folder as the notebooks:

| File | Dataset | Source |
| --- | --- | --- |
| `insurance.csv` | Medical Cost Personal Dataset | Kaggle |
| `heart.csv` | Heart Disease dataset | Kaggle / UCI |

The notebooks load them with `pd.read_csv("insurance.csv")` and `pd.read_csv("heart.csv")`. If you keep the files elsewhere, update those paths.

### 4. Run

```bash
jupyter notebook
```

Open any notebook and run the cells from top to bottom.

---

## Tech Stack

Python · pandas · seaborn · matplotlib · scikit-learn · Jupyter Notebook

---

## Author

**Junaid Ashraf**

- GitHub: [@junaid0994](https://github.com/junaid0994)
- LinkedIn: [junaid-ashraf](https://www.linkedin.com/in/junaid-ashraf-53123b37a/)

If you find this repo useful, consider giving it a star.
