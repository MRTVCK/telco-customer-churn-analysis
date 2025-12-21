# 📊 Telecom Customer Churn Analysis & Prediction

## Overview
This project analyzes customer churn behavior for a telecommunications company and builds a churn prediction model to identify high-risk customers. The goal is to combine **data analysis, machine learning, and business-focused visualization** to support retention strategies.

The project includes:
- Exploratory data analysis (EDA)
- Feature engineering
- Churn prediction modeling
- Model explainability
- Interactive Tableau dashboards for stakeholder insights

---

## 🧠 Problem Statement
Customer churn is costly for telecom companies. Identifying which customers are likely to churn — and *why* — enables targeted retention efforts, optimized pricing, and improved customer experience.

This project answers:
- Which customer segments churn the most?
- How does churn vary by contract type and tenure?
- Which customers are at highest risk of churn?
- What features are most influential in churn prediction?

---

## 🛠️ Tools & Technologies
- **Python**: pandas, numpy, scikit-learn
- **Machine Learning**: Logistic Regression, Random Forest
- **Visualization**: Tableau Public
- **Environment**: Python virtual environment (venv)

---

## 📁 Project Structure
telco-customer-churn-analysis/
│
├── data/
│ ├── raw/ # Original dataset
│ └── processed/ # Cleaned and engineered data
│
├── notebooks/
│ ├── 01_eda.ipynb
│ ├── 02_feature_engineering.ipynb
│ └── 03_churn_model.ipynb
│
├── outputs/
│ ├── telco_churn_predictions_for_tableau.csv
│ └── telco_churn_feature_importance.csv
│
├── tableau/
│ └── dashboard_screenshots.png
│
└── README.md


---

## 📊 Exploratory Data Analysis
Key findings from EDA:
- Month-to-month contracts show significantly higher churn rates
- Short-tenure customers churn more frequently than long-tenure customers
- Higher monthly charges correlate with increased churn risk
- Electronic check payment method is associated with higher churn

---

## 🤖 Machine Learning Models
Two models were trained and evaluated:

### Logistic Regression
- ROC AUC: **~0.85**
- Interpretable coefficients used for feature importance
- Used for probability-based churn predictions

### Random Forest
- ROC AUC: **~0.82**
- Captures non-linear relationships
- Used for performance comparison

Logistic Regression was selected for final predictions due to its balance of performance and explainability.

---

## 🔍 Model Outputs
The final dataset includes:
- **Predicted Churn Label** (0 / 1)
- **Predicted Churn Probability**
- Customer-level features for visualization and analysis

A separate feature importance file highlights the strongest churn drivers using absolute model coefficients.

---

## 📈 Tableau Dashboards
Interactive dashboards were built in Tableau Public to translate model results into actionable insights:

### Dashboards Include:
- **Churn by Contract Type**
- **Churn by Tenure Bucket**
- **High-Risk Customer Table** (ranked by churn probability)
- **Feature Importance Bar Chart**
- **KPI Tiles** (overall churn rate)

🔗 **Tableau Public Link:**  
(https://public.tableau.com/app/profile/destin.tucker/viz/TelecomCustomerChurnAnalysisPredictionDashboard/DashboardChurnOverview?publish=yes)

https://public.tableau.com/app/profile/destin.tucker/viz/TelecomCustomerChurnAnalysisPredictionDashboard/RetentionActionDashboard?publish=yes
---

## 💡 Business Insights
- Month-to-month customers represent the highest churn risk
- Long-term contracts significantly reduce churn probability
- Early-tenure customers should be targeted with retention incentives
- High monthly charges combined with short tenure indicate elevated risk

---

## 🚀 Future Improvements
- Hyperparameter tuning
- Class imbalance handling (SMOTE)
- SHAP values for deeper model explainability
- Deployment as a web dashboard or internal tool

---

## 👤 Author
**Destin Tucker**

- GitHub: https://github.com/MRTVCK 
- Tableau Public: https://public.tableau.com/app/profile/destin.tucker/vizzes

---

## 📌 Notes
This project was created for learning and portfolio purposes only...
