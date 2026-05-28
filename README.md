# Enterprise Customer Retention Engine & Production API Gateway

A production-grade machine learning microservice architecture that combines an optimized Random Forest classifier with a live FastAPI backend framework to serve real-time customer churn predictions.

## 🛠️ Updated Tech Stack
* **Languages & Core:** Python, SQL
* **Data Engineering & Analytics:** Pandas, NumPy, Imbalanced-learn
* **Machine Learning & Modeling:** Scikit-learn (Logistic Regression, Random Forest Ensembles)
* **Production Serving & MLOps:** FastAPI, Uvicorn, Pydantic, REST API Architectures, Git

## 📈 Model Performance & Business Logic
* **Exploratory Data Analysis (EDA):** Identified that low-tenure customer accounts and high monthly service charges are the primary driving indicators for corporate revenue leaks.
* **Model Benchmark Evaluations:**
  * Baseline Logistic Regression: **82.00% Accuracy**
  * Production-Grade Random Forest Classifier: **85.00% Accuracy** (Optimized for operational Recall to minimize missed churn indicators).

## 📂 Enterprise Project Architecture
Unlike standard academic or course assignments that rely on loose, manual Jupyter Notebook execution (.ipynb), this system is built using professional, modular software engineering practices:
```text
📂 customer-churn-prediction
 ├── 📂 src/
 │    ├── __init__.py          # Defines directory as a Python package module
 │    ├── data_processing.py   # Handles automated data pipelines and encoding transformations
 │    └── train.py             # Executes training pipelines and saves model binaries
 ├── app.py                    # Production FastAPI gateway server serving the prediction API
 └── requirements.txt          # Explicit production library dependencies
```

## 💻 Local Workspace Execution Setup

1. **Initialize the virtual environment dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute the automated machine learning engine to process logs and save the model binary:**
   ```bash
   python src/train.py
   ```

3. **Launch the live local Uvicorn deployment server channel backend:**
   ```bash
   uvicorn app:app --reload
   ```

4. **Interact with the API Endpoint System:**
   Open your browser and navigate to the interactive Swagger UI gateway: `http://127.0.0` to test raw payload queries against the live model.

