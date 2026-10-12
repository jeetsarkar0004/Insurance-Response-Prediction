# Insurance Customer Response Prediction

Built a predictive classification engine to forecast customer responses to insurance offers, enabling high-precision targeted marketing and optimized sales channel allocation.


## Project at a glance

| | |
|---|---|
| **Task** | Binary classification (`Response`: 1 = interested, 0 = not interested) |
| **Data** | 381,109 customers, 12 columns, no missing values or duplicates |
| **Class balance** | Only ~12.3% say yes (imbalanced) |
| **Models** | Logistic Regression, Decision Tree, Random Forest |
| **Imbalance handling** | SMOTE inside an `imblearn` Pipeline (resampling happens only on training folds, so no leakage) |
| **Tuning** | GridSearchCV (Logistic Regression), RandomizedSearchCV (trees) |
| **Explainability** | SHAP (`TreeExplainer`) |
| **Charts** | Plotly |

## Results (held-out test set, default 0.5 threshold)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.695 | 0.276 | 0.916 | 0.424 | **0.846** |
| Decision Tree | 0.755 | 0.307 | 0.791 | 0.442 | 0.829 |
| **Random Forest** (selected) | **0.805** | **0.328** | 0.564 | 0.415 | 0.840 |

Random Forest was selected as the final model because its ROC-AUC is within about 0.006 of the best, it has the best precision and accuracy (it contacts roughly half as many customers as Logistic Regression), and it supports exact SHAP explanations. Logistic Regression is a near-equal, simpler alternative. Precision for the "yes" class is modest (about 1 in 3 flagged customers accepts), so the model is best used to **rank and prioritise** customers, not as a yes/no oracle.

## Key findings

- **Prior vehicle damage + not already insured** is the highest-converting segment (~25% respond vs ~0.5% for everyone else).
- **Already-insured customers** almost never respond, so de-prioritise them.
- **Ages 30-49** respond at ~20%, versus ~4% for under-30s.
- `Policy_Sales_Channel` matters, but partly because it is correlated with age.

The notebook loads `data.csv` using a relative path, so keep the notebook and the CSV in the same folder. Running it end to end retrains the models (the hyperparameter searches can take a while) and writes `random_forest_model.joblib` and `feature_columns.json` next to the notebook. The `.joblib` file is git-ignored because it can be large.

## Data

Each row is a customer: `id`, `Gender`, `Age`, `Driving_License`, `Region_Code`, `Previously_Insured`, `Vehicle_Age`, `Vehicle_Damage`, `Annual_Premium`, `Policy_Sales_Channel`, `Vintage` (days as a customer) and the target `Response`.

## Possible next steps

- Try gradient-boosted models (XGBoost / LightGBM) and a larger hyperparameter search.
- Use `SMOTENC` or class weights instead of plain SMOTE (plain SMOTE creates fractional values in code and binary columns).
- Tune the decision threshold to the campaign's cost per contact versus value per conversion.
- Retrain periodically as new response data arrives.

