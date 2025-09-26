# 🛒 Sales Forecasting Project  

## 📌 Overview  
This project implements an **end-to-end Machine Learning pipeline** for forecasting **weekly sales** across stores and departments.  
The pipeline is modular, production-ready, and follows clean architecture principles:  
- **Data Ingestion**  
- **Data Transformation (feature engineering & scaling)**  
- **Model Training & Selection**  
- (Future: Model Evaluation, Prediction Pipeline, Deployment)

The goal is to predict `Weekly_Sales` more accurately by leveraging historical sales data, holidays, promotions, and store-level features.

---

## 📂 Project Structure  

---

## ⚙️ Features  

- **Data Ingestion**  
  - Reads raw CSV files.  
  - Splits into train/test.  

- **Data Transformation**  
  - Handles feature engineering (holiday flags, markdown sums, etc.).  
  - Missing value treatment.  
  - Label encoding and scaling.  
  - Saves `preprocessor.pkl` for later inference.  

- **Model Training**  
  - Supports multiple models (Random Forest, XGBoost, LightGBM, CatBoost, etc.).  
  - Evaluates models using RMSE (and Weighted MAE for holiday weeks).  
  - Automatically selects the best model.  
  - Saves final trained model in `artifacts/model.pkl`.  

- **Logging & Exception Handling**  
  - Centralized logger (`logger.py`).  
  - Custom exception handler (`exception.py`).  

---



