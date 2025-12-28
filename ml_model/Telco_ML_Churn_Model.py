import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.impute import SimpleImputer

# 1. Load data
df = pd.read_csv("v_churn_features.csv")

# 2. Target + features
y = df["churn"].astype(int)          # True/False → 1/0
X = df.drop(columns=["churn", "customer_id"])

# 3. Identify column types
numeric_cols = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "avg_monthly_revenue",
    "addon_services_count"
]

categorical_cols = [col for col in X.columns if col not in numeric_cols]

# 4. Preprocessing with imputers
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# 5a. Logistic Regression model
log_reg = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
y_proba = log_reg.predict_proba(X_test)[:, 1]

print("LogReg AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))

# 5b. RandomForest (optional second model)
rf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42
    ))
])

rf.fit(X_train, y_train)
rf_proba = rf.predict_proba(X_test)[:, 1]
print("RandomForest AUC:", roc_auc_score(y_test, rf_proba))

# 6. Use the BEST model (LogReg) to generate predictions for ALL rows
log_reg.fit(X, y)  # fit on full data for final predictions

# Predicted probability of churn (class 1)
y_proba_all = log_reg.predict_proba(X)[:, 1]
# Predicted label (0 or 1) using default 0.5 threshold
y_pred_all = (y_proba_all >= 0.5).astype(int)

# 7. Build a DataFrame to export to Tableau
export_df = pd.DataFrame({
    "customer_id": df["customer_id"],
    "actual_churn": y.values,           # 0 or 1
    "predicted_churn_label": y_pred_all,
    "predicted_churn_proba": y_proba_all,
})

# Add a few important original columns for easier Tableau viz
cols_to_keep = [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "tenure_months",
    "contract",
    "paperless_billing",
    "payment_method",
    "monthly_charges",
    "total_charges",
    "avg_monthly_revenue",
    "addon_services_count",
]

for col in cols_to_keep:
    export_df[col] = df[col]

# 8. Save predictions CSV
export_df.to_csv("telco_churn_predictions_for_tableau.csv", index=False)
print("Saved: telco_churn_predictions_for_tableau.csv")

# 9. Get feature importances for Logistic Regression
#    We need to grab the final trained model and the transformed feature names
final_log_reg_model = log_reg.named_steps["model"]
final_preprocess = log_reg.named_steps["preprocess"]

feature_names = final_preprocess.get_feature_names_out()
coefficients = final_log_reg_model.coef_.flatten()

feat_imp_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients,
    "abs_coefficient": np.abs(coefficients)
}).sort_values("abs_coefficient", ascending=False)

feat_imp_df.to_csv("telco_churn_feature_importance.csv", index=False)
print("Saved: telco_churn_feature_importance.csv")