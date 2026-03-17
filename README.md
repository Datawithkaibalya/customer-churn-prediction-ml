# Customer Churn Prediction (Machine Learning)

pandas
numpy
scikit-learn
matplotlib
seaborn
imbalanced-learn

## 📊 Problem Statement
Predict whether a customer will churn based on historical data.

## 🛠️ Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

## ⚙️ Steps
- Data Cleaning & Preprocessing
- Feature Encoding & Scaling
- Model Building (Logistic Regression, Random Forest)
- Model Evaluation

## 📈 Results
- Achieved ~82% accuracy
- Identified key churn factors

## 🚀 Conclusion
Helps businesses identify high-risk customers and improve retention.


import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("data.csv")

# Basic EDA
print(df.head())
print(df.info())

# Preprocessing
df = df.dropna()

# Feature & Target
X = df.drop("target", axis=1)
y = df["target"]

# Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))

## 📊 Model Comparison

| Model                | Accuracy |
|---------------------|----------|
| Logistic Regression | 82%      |
| Random Forest       | 85%      |

## 🔍 Key Insights
- Customers with low tenure are more likely to churn
- High monthly charges increase churn probability
- 
