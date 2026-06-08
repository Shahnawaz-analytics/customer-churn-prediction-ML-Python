# 📡 Customer Churn Prediction — Telecom

A machine learning project that predicts which telecom customers are likely to cancel their subscription, enabling the business to take proactive retention action before it's too late.

---

## 🧩 Problem Statement

Customer churn is one of the most expensive problems in the telecom industry. Acquiring a new customer costs 5–7x more than retaining an existing one. This project builds a classification model to identify at-risk customers **before** they leave, so the business can intervene with targeted offers or support.

---

## 📊 Dataset

- **Source:** IBM Telco Customer Churn Dataset
- **Size:** 7,043 customers × 21 features
- **Target:** `Churn` — Yes (churned) / No (stayed)
- **Class Distribution:** ~73.5% No Churn / ~26.5% Churn *(imbalanced)*

**Key Features Include:**
- Demographics — gender, senior citizen, partner, dependents
- Services — phone, internet, streaming, security, tech support
- Account — contract type, payment method, paperless billing
- Charges — monthly charges, total charges, tenure

---

## 🔧 Tech Stack

| Area | Tools |
|---|---|
| Language | Python 3 |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Model Saving | Joblib |
| Web App | Streamlit |

---

## 🚀 Project Workflow

### 1. Data Cleaning
- Converted `TotalCharges` from string to numeric (contained hidden spaces)
- Filled 11 missing values in `TotalCharges` with median
- Dropped `customerID` (irrelevant to prediction)

### 2. Exploratory Data Analysis (EDA)
- Visualized churn distribution and confirmed class imbalance (26.5% churn)
- Found month-to-month contract customers churn ~3x more than two-year customers
- Plotted distributions of Tenure, MonthlyCharges, TotalCharges
- Correlation heatmap showed TotalCharges highly correlated with Tenure

### 3. Preprocessing
- Applied `pd.get_dummies()` for One-Hot Encoding of all categorical features
- Used `StandardScaler` for numerical feature normalization
- Addressed class imbalance using `class_weight='balanced'` — preventing the model from simply predicting "No Churn" every time

### 4. Model Building & Comparison

| Model | Accuracy | Churn Recall |
|---|---|---|
| Logistic Regression | 74.7% | 82% |
| Random Forest (baseline) | ~79% | ~76% |
| **Tuned Random Forest** | **~80%** | **85%** |

### 5. Hyperparameter Tuning
Used `GridSearchCV` with 5-fold cross-validation, optimizing for **Recall** (not accuracy) because missing a churner costs more than a false alarm.

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}
```

### 6. Evaluation
- **ROC-AUC Score: 0.86** — excellent class separation
- Plotted ROC Curve and Confusion Matrix
- 5-Fold Cross-Validation for reliable performance estimate

---

## 📈 Key Results

- ✅ Final model: **Tuned Random Forest**
- ✅ ROC-AUC: **0.86**
- ✅ Churn Recall: **85%** — model correctly identifies 85% of actual churners

**Top 5 Features Driving Churn:**
1. Tenure (longer = less likely to churn)
2. Total Charges
3. Monthly Charges
4. Contract Type — Month-to-Month customers churn most
5. Internet Service — Fiber Optic users show higher churn

---

## 💡 Business Recommendations

1. **Incentivize long-term contracts** — offer discounts to move month-to-month users to 1 or 2-year plans
2. **High-value customer monitoring** — proactively reach out to customers with high monthly charges
3. **Tenure-based loyalty rewards** — implement offers at key tenure milestones where churn risk spikes
4. **Fiber Optic service audit** — investigate whether high churn in Fiber Optic is a pricing or quality issue
5. **Predictive retention team** — use model scores to flag high-risk customers for the customer success team

---

## 🖥️ Streamlit Web App

Built an interactive web app where anyone can enter customer details and get an instant churn prediction with probability score and suggested retention actions.


👉 [Click here to try the app]
([https://your-link.streamlit.app](https://customer-churn-prediction-ml-python-h2kkscjiicwytqqe7dwpdt.streamlit.app/))

---

## 📁 Project Structure

```
customer-churn-prediction-ML-Python/
│
├── app.py                                      # Streamlit web application
├── Customer_Churn_Prediction_Improved.ipynb    # Full ML notebook
├── customer_churn_model.pkl                    # Saved trained model
├── scaler.pkl                                  # Saved StandardScaler
├── requirements.txt                            # Python dependencies
└── README.md                                   # Project documentation
```

---

## 📬 Connect

**GitHub:** [Shahnawaz-analytics](https://github.com/Shahnawaz-analytics)

**Linkedin:** [Shahnawaz-khan](https://www.linkedin.com/in/shahnawazkhan09/)

