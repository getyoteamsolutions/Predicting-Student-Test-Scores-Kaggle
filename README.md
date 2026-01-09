# Linear Regression + Residual Boosting with LightGBM

This repository implements a **two-stage regression modeling approach** that combines **Linear Regression** with **LightGBM** using **residual learning**.  
The goal is to leverage the strengths of both linear and tree-based models to achieve better predictive performance.

This approach is commonly used in **Kaggle competitions** and **real-world machine learning systems**.

---

## Problem Statement

Predict a continuous target variable using structured tabular data containing both numerical and categorical features.

---

## Modeling Strategy

### Stage 1: Linear Regression (Baseline Model)

- Train a Linear Regression model using **K-Fold cross-validation**
- Generate:
  - **Out-of-Fold (OOF) predictions** for training data
  - **Averaged predictions** for test data
- Captures global **linear relationships** in the data

---

### Stage 2: Residual Learning with LightGBM

- Compute residuals:
  ```
  residual = y_true − y_pred_LR
  ```
- Train a **LightGBM Regressor** on:
  - Same encoded features
  - Target = residuals
- Learns **non-linear patterns** missed by Linear Regression

---

### Final Prediction

```
final_prediction = LR_prediction + LGBM_residual_prediction
```

---

## Why This Approach Works

- Linear Regression efficiently models global trends
- LightGBM captures complex non-linear interactions
- Residual learning avoids redundancy and overfitting
- Improves generalization compared to a single model

---

## Feature Engineering & Encoding

- Categorical features are encoded using **one-hot encoding** (`pd.get_dummies`)
- Train and test feature matrices are aligned to ensure identical columns
- All features passed to models are of type:
  - `float`
  - `int`
  - `bool`
- No raw categorical (`object`) columns are used

---

## Cross-Validation

- **K-Fold Cross-Validation**
- No data leakage:
  - Validation predictions come from unseen folds
  - Test predictions are averaged across folds
- OOF predictions are used to:
  - Evaluate model performance
  - Compute residuals safely

---

## Evaluation Metric

- **RMSE (Root Mean Squared Error)**
- RMSE is computed on:
  - Linear Regression OOF predictions
  - Residual LightGBM OOF predictions
  - (Optionally) combined final predictions

---

## Project Structure

```
├── lr_residual_model_lgbm.ipynb   # Main notebook
├── README.md                      # Project documentation
```

---

## Libraries Used

- Python
- NumPy
- Pandas
- scikit-learn
- LightGBM

---

## Key Highlights

- Demonstrates correct use of:
  - Out-of-Fold predictions
  - Residual modeling
  - Model stacking concepts
- Avoids common LightGBM issues with categorical data
- Clean, reproducible, and Kaggle-ready workflow
- Easily extensible to more advanced ensembles

---

## Possible Improvements

- Add early stopping for LightGBM
- Try Ridge / Lasso / ElasticNet as base model
- Compare against a pure LightGBM model
- Add feature importance analysis
- Hyperparameter tuning with Optuna or GridSearch

---

## Author Notes

This project was built as a **learning-focused, competition-style implementation** to deeply understand:
- Cross-validation mechanics
- Residual boosting
- Hybrid model design

---
 
