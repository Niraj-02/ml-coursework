# Machine Learning Coursework

Three end-to-end supervised learning projects from Kaggle assignments in the *Machine Learning Practice* course of the IIT Madras BS in Data Science programme (2025). Each one goes through the full workflow: data cleaning, EDA, preprocessing, training many models, hyperparameter tuning and a final submission.

| # | Project | Task | Best model | Validation result |
|---|---|---|---|---|
| 1 | [Flight price prediction](01-flight-price-prediction) | Regression | Gradient Boosting (tuned) | **R² = 0.971** |
| 2 | [Customer churn prediction](02-customer-churn-prediction) | Binary classification (imbalanced) | XGBoost / LightGBM (tuned) | **ROC AUC = 0.876** |
| 3 | [Restaurant rating prediction](03-restaurant-rating-prediction) | 5-class classification from text + tabular data | XGBoost / LightGBM on sentence embeddings | **Accuracy = 0.711** |

---

## 1. Flight price prediction
Predict ticket price from airline, route, departure/arrival time slots, stops, class, duration and days until departure.

- Cleaned mislabelled flight codes, imputed missing values, removed duplicates and capped outliers
- EDA: price is right-skewed; business vs. economy class is the biggest driver; duration correlates positively with price and days-left negatively
- Compared **13 regressors**: Linear, Ridge, Lasso, SVR, KNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost and tuned variants
- Tree ensembles beat linear models (R² 0.97 vs. 0.91), which shows strong non-linear interactions such as class × airline

## 2. Customer churn prediction
Predict whether a bank customer will leave, from demographics, balance, tenure, product count and activity.

- Median/mode imputation, IQR capping of outliers, dropped identifier columns
- EDA: churn rises sharply after age 45, is higher among female customers, and varies by country
- Compared **11 classifiers**, from Logistic Regression and KNN to Random Forest, ExtraTrees, XGBoost, LightGBM and CatBoost
- Tracked **F1 and ROC AUC** rather than accuracy alone, because the classes are imbalanced. Gradient-boosted trees led on both (F1 ≈ 0.61, AUC ≈ 0.876)

## 3. Restaurant rating prediction
Predict a customer's 1–5 star rating from the review text plus store metadata (location, rating count, review age).

- Turned free-text review age ("2 years ago", "a month ago") into a numeric number of days
- Started with TF-IDF (5,000 features, slow to train), then switched to dense **sentence embeddings** (`all-mpnet-base-v2`, GPU-encoded) combined with the scaled tabular features
- Ratings are bimodal (mostly 1★ and 5★), so the middle classes are the hard part
- Compared 7 classifiers and tuned 4. The final submission is a **soft-voting ensemble**

---

## Structure

```
.
├── 01-flight-price-prediction/
├── 02-customer-churn-prediction/
├── 03-restaurant-rating-prediction/
└── README.md
```

Each folder contains the full notebook with outputs. The competition data can't be redistributed, so the notebooks expect it under `/kaggle/input/...`.

**Stack:** pandas, NumPy, scikit-learn, XGBoost, LightGBM, CatBoost, sentence-transformers, matplotlib, seaborn
