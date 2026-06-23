# 💳 Credit Card Fraud Detection System

An end-to-end machine learning project for detecting fraudulent credit card transactions using advanced feature engineering, threshold optimization, and a deployed interactive Streamlit application.

---

## 📌 Overview

Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions represent only a tiny fraction of total transactions.

This project focuses on:

- Detecting fraud with **high recall**
- Reducing false negatives (missed fraud)
- Building a deployable inference pipeline
- Preventing data leakage
- Optimizing decision thresholds for business needs

---

## 📂 Project Structure

```text
Credit-Card-Fraud-Detection/
│── app.py
│── requirements.txt
│── README.md
│── .gitignore
│── config.json
│── model.pkl
│── scaler.pkl
│
├── data/
│   └── sample_data.csv
│
├── notebooks/
│   └── credit_card_fraud_analysis.ipynb
│
├── plots/
    ├── eda_plots.png
    ├── threshold_plots.png
 
```

---

## 📊 Dataset

Dataset used: **Credit Card Fraud Detection Dataset (Kaggle)**

Contains:

- **284,807 transactions**
- **492 fraud cases**
- Highly imbalanced dataset (~0.172% fraud)

Features:

- `Time`
- `Amount`
- `V1–V28` (PCA-anonymized features)
- `Class` (Target)

---

## ⚙️ Feature Engineering

This project uses custom engineered features to improve fraud detection.

### Amount-based features

- `log_amount`
- `amount_log_sq`
- `is_round_amount`
- `is_small_amount`
- `amount_to_mean`
- `is_high_amount`

### Time-based features

- `Hour`
- `hour_of_day`
- `hour_sin`
- `hour_cos`
- `is_night`

### Aggregated V-feature statistics

- `v_sum_sq`
- `v_max_abs`

Total engineered features: **43**

---

## 🤖 Model Pipeline

### Data preprocessing

- Train-test split
- Robust scaling
- Feature ordering preservation
- Leakage-safe train statistics

### Models experimented

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

### Hyperparameter optimization

Implemented using **Optuna**

Optimization goal:

- Maximize recall
- Maintain minimum precision

---

## 🏆 Final Model

**XGBoost Classifier**

Threshold tuned for fraud detection:

```text
0.0062
```

Reason:

Fraud detection prioritizes **high recall over precision**.

---

## 📈 Model Performance

### Default Threshold (0.5)

| Metric | Fraud |
|---|---:|
| Precision | 0.95 |
| Recall | 0.77 |
| F1-score | 0.85 |

---

### Tuned Threshold (0.0062)

| Metric | Fraud |
|---|---:|
| Precision | 0.55 |
| Recall | 0.85 |
| F1-score | 0.67 |

---

## 💼 Business Interpretation

Threshold tuning improves fraud capture:

- Catches more fraudulent transactions
- Reduces financial losses
- Increases false positives (acceptable in fraud systems)

Tradeoff:

- **Higher recall = fewer missed frauds**
- **Lower precision = more manual reviews**

---

## 🚀 Streamlit Deployment

Interactive app supports:

### Single Transaction Scoring

Input:

- Time
- Amount
- V1–V28

Output:

- Fraud probability
- Final classification

---

### Batch CSV Scoring

Upload transaction CSV and get:

- Fraud probability per transaction
- Prediction labels
- Downloadable results

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Optuna

---

## ▶️ How to Run

Clone repository:

```bash
git clone https://github.com/Subhanu06/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
streamlit run app.py
```

---

## 📚 Key Learnings

- Handling extreme class imbalance
- Threshold optimization in fraud detection
- Precision-recall tradeoffs
- Preventing train-test leakage
- Building reproducible ML pipelines
- Deploying ML systems

---

## 🔮 Future Improvements

- SHAP explainability
- Real-time API deployment
- Ensemble stacking
- Drift monitoring
- Fraud severity tiers
- Automated retraining

---

## 👨‍💻 Author

**Subhanu Dhar**

GitHub: https://github.com/Subhanu06
