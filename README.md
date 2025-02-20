# Credit Card Customer Churn - EDA & Modelling

## 📊 Overview
This project focuses on analyzing and predicting customer churn using the **Credit Card Customers** dataset. The goal is to identify key factors leading to customer churn and build a predictive model to help financial institutions retain valuable customers.

## 🎯 Goals
1. **Exploratory Data Analysis (EDA):**
   - Understand customer behavior and differentiate between churning and non-churning profiles.
   - Identify patterns, trends, and anomalies that influence customer churn.

2. **Churn Prediction Modeling:**
   - Build a machine learning model to predict whether a customer will churn.
   - Optimize the model with a focus on the **Recall** metric (target > 62%) to minimize false negatives.

## 🛠️ Libraries Used
- **Pandas** – Data manipulation
- **NumPy** – Numerical operations
- **Matplotlib** & **Seaborn** – Data visualization
- **Scikit-learn** – Machine learning models & preprocessing
- **XGBoost** – Advanced boosting model
- **Imbalanced-learn** – Handling class imbalance
- **SHAP** – Feature importance explanation

## 📁 The Data
- **Source:** Credit Card Customers dataset
- **Target Variable:** `Attrition_Flag` (Churned or Non-Churned)
- **Features Include:**
  - Customer demographics
  - Credit card usage patterns
  - Account details

### 🔄 Data Preprocessing
- Handled missing values
- Encoded categorical features
- Scaled numerical data
- Managed class imbalance using SMOTE

## 📊 Customer Profiles
### 1. **Exploratory Data Analysis:**
   - Uncovered customer behavior trends
   - Identified high-risk profiles likely to churn

### 2. **Churn vs. Non-Churn Profiles:**
   - Compared transaction counts, credit utilization, and customer demographics
   - Analyzed correlations between features and churn likelihood

## 🤖 Customer Churn Prediction
### 📂 Data Preparation
- Split data into training and testing sets
- Applied feature scaling and encoding

### 🏆 Model Training
- Trained multiple classifiers: **Logistic Regression, Random Forest, XGBoost**
- Evaluated models primarily on **Recall**

### 📈 Model Evaluation
- Metrics Used: **Accuracy, Precision, Recall, F1-Score, ROC-AUC**
- Focused on improving Recall while maintaining overall model performance

### ⚙️ Hyperparameter Tuning
- Used **GridSearchCV** for model optimization
- Fine-tuned hyperparameters for best Recall score

### 📌 Feature Importance
- Applied SHAP values to interpret the most influential features impacting churn

## ✅ Conclusion
- Identified key drivers of customer churn
- Built a predictive model achieving a Recall above the 62% target
- Provided actionable insights for customer retention strategies

## 💡 Future Work
- Incorporate more customer behavior data for deeper insights
- Explore deep learning models for improved accuracy
- Implement real-time churn prediction in business applications

## 📬 Contact
For any questions or feedback, feel free to reach out! 🚀
