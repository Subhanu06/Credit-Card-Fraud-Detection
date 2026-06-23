# 💳 Credit Card Fraud Detection System

An end-to-end Machine Learning system for detecting fraudulent credit card transactions using advanced feature engineering, class imbalance handling, threshold optimization, and an interactive Streamlit web application.

---

## 🚀 Live Demo

🔗 **Live Application:**
https://credit-card-fraud-detection2026.streamlit.app/

📂 **Source Code:**
https://github.com/Subhanu06/Credit-Card-Fraud-Detection

---

## 📌 Project Overview

Credit card fraud detection is one of the most challenging classification problems in machine learning due to the extreme imbalance between legitimate and fraudulent transactions.

This project focuses on:

* Detecting fraudulent transactions with high recall
* Minimizing costly false negatives
* Building a production-ready ML inference pipeline
* Preventing data leakage during training
* Optimizing classification thresholds for business requirements
* Deploying the model through an interactive web application

---

## 📂 Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── config.json
├── model.pkl
├── scaler.pkl
│
├── data/
│   └── sample_data.csv
│
├── notebooks/
│   └── credit_card_fraud_analysis.ipynb
│
├── plots/
    ├── eda_plots.png
    └── threshold_plots.png
```

---

## 📊 Dataset

**Dataset:** Credit Card Fraud Detection Dataset (Kaggle)

### Dataset Statistics

| Metric                  | Value   |
| ----------------------- | ------- |
| Total Transactions      | 284,807 |
| Fraudulent Transactions | 492     |
| Legitimate Transactions | 284,315 |
| Fraud Percentage        | 0.172%  |

### Features

* Time
* Amount
* V1 – V28 (PCA-transformed confidential features)
* Class (Target Variable)

The severe class imbalance makes accuracy an unreliable metric and requires specialized evaluation techniques.

---

## ⚙️ Feature Engineering

Several domain-inspired features were engineered to improve fraud detection performance.

### Amount-Based Features

* log_amount
* amount_log_sq
* is_round_amount
* is_small_amount
* amount_to_mean
* is_high_amount

### Time-Based Features

* Hour
* hour_of_day
* hour_sin
* hour_cos
* is_night

### Aggregated Statistical Features

* v_sum_sq
* v_max_abs

### Total Features

* Original Features: 30
* Engineered Features: 13
* Final Features Used: 43

---

## 🤖 Machine Learning Pipeline

### Data Preprocessing

* Missing value validation
* Train-test split
* RobustScaler normalization
* Feature order preservation
* Leakage-safe feature generation
* Statistical feature transformation

### Models Evaluated

* Logistic Regression
* Random Forest Classifier
* XGBoost Classifier
* LightGBM Classifier

### Hyperparameter Optimization

Hyperparameter tuning was performed using Optuna.

Optimization Objectives:

* Maximize Recall
* Maintain acceptable Precision
* Improve Fraud Detection Rate

---

## 🏆 Final Model

### Selected Model

XGBoost Classifier

### Optimized Decision Threshold

```text
0.0062
```

Instead of using the default threshold of 0.5, the classification threshold was optimized to maximize fraud detection performance.

This approach prioritizes:

* Higher Recall
* Lower Missed Fraud Cases
* Better Business Protection

---

## 📈 Model Performance

### Default Threshold (0.5)

| Metric    | Fraud Class |
| --------- | ----------: |
| Precision |        0.95 |
| Recall    |        0.77 |
| F1 Score  |        0.85 |

### Tuned Threshold (0.0062)

| Metric    | Fraud Class |
| --------- | ----------: |
| Precision |        0.55 |
| Recall    |        0.85 |
| F1 Score  |        0.67 |

### Performance Insight

Threshold tuning significantly improves fraud detection capability by increasing recall and reducing missed fraudulent transactions.

---

## 💼 Business Impact

Fraud detection systems generally prioritize recall over precision because missing a fraudulent transaction can be far more expensive than investigating a legitimate one.

### Benefits

* Increased fraud capture rate
* Reduced financial losses
* Improved customer protection
* Better risk management

### Trade-Off

* Higher Recall → Fewer missed frauds
* Lower Precision → More manual reviews

This trade-off is acceptable in many real-world fraud detection systems.

---

## 🌐 Streamlit Application

The project includes a fully deployed interactive web application.

### Single Transaction Prediction

Users can manually enter:

* Time
* Amount
* V1–V28 Features

The application returns:

* Fraud Probability
* Fraud Prediction
* Confidence Score

### Batch CSV Prediction

Users can upload transaction datasets and receive:

* Fraud probabilities
* Predicted labels
* Downloadable prediction results

---

## 🛠 Technology Stack

### Programming

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-Learn
* XGBoost
* LightGBM
* Optuna

### Deployment

* Streamlit
* GitHub

---

## ▶️ Installation & Usage

### Clone Repository

```bash
git clone https://github.com/Subhanu06/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

### Open Browser

```text
http://localhost:8501
```

---

## 📚 Key Learnings

Through this project, I gained hands-on experience with:

* Extreme class imbalance handling
* Feature engineering techniques
* Threshold optimization
* Precision-recall trade-offs
* Model evaluation for imbalanced datasets
* Leakage prevention
* Hyperparameter tuning with Optuna
* End-to-end ML deployment using Streamlit

---

## 🔮 Future Enhancements

* SHAP Explainability Dashboard
* REST API Deployment
* Real-Time Fraud Monitoring
* Model Drift Detection
* Ensemble Learning Techniques
* Automated Model Retraining Pipeline
* Cloud Deployment with Docker

---

## 👨‍💻 Author

### Subhanu Dhar

GitHub: https://github.com/Subhanu06

Project Repository:
https://github.com/Subhanu06/Credit-Card-Fraud-Detection

Live Application:
https://credit-card-fraud-detection2026.streamlit.app/

---

⭐ If you found this project useful, consider giving the repository a star.
