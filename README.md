# Customer Churn Prediction Project

## 🔍 Overview

This project focuses on predicting customer churn using transactional data from an online retail business. The goal is to identify customers who are likely to stop purchasing, enabling proactive retention strategies.

The dataset used in this project comes from the UCI Machine Learning Repository:
[https://archive-beta.ics.uci.edu/dataset/502/online%2Bretail%2Bii?utm_source=chatgpt.com](https://archive-beta.ics.uci.edu/dataset/502/online%2Bretail%2Bii?utm_source=chatgpt.com)

## 📊 Dataset Description

The dataset contains transactional records for an online retail store. Each row represents a single transaction line (an item within an invoice).

### Columns

* **InvoiceNo**: Unique identifier for each transaction (invoice). Some invoices starting with 'C' indicate cancellations.
* **StockCode**: Product/item code.
* **Description**: Name/description of the product.
* **Quantity**: Number of items purchased in the transaction.
* **InvoiceDate**: Date and time of the transaction.
* **UnitPrice**: Price per item.
* **CustomerID**: Unique identifier for each customer.
* **Country**: Country of the customer.

### Example

| InvoiceNo | StockCode | Description         | Quantity | InvoiceDate      | UnitPrice | CustomerID | Country |
| --------- | --------- | ------------------- | -------- | ---------------- | --------- | ---------- | ------- |
| 536365    | 85123A    | WHITE HANGING HEART | 6        | 2010-12-01 08:26 | 2.55      | 17850      | UK      |
| 536365    | 71053     | WHITE METAL LANTERN | 6        | 2010-12-01 08:26 | 3.39      | 17850      | UK      |

## ⚙️ Data Processing

Churn was defined as **customer inactivity over the last couple of months** (i.e., no purchases in a defined recent period).

## 🧩 Feature Engineering

Several behavioral features were extracted to describe customer activity.


| CustomerID | Recency (days) | Frequency | Monetary | Returns | Multiple Purchases | Avg Days Between Purchases | Churn |
| ---------- | -------------- | --------- | -------- | ------- | ------------------ |----------------------------| ----- |
| 12345      | 12             | 5         | 350.75   | 1       | 1                  | 18.2                       | 0     |
| 12346      | 95             | 1         | 89.00    | 0       | 0                  | 95.0                       | 1     |
| 12347      | 30             | 3         | 220.40   | 0       | 1                  | 25.5                       | 0     |

### 1. RFM Features

* **Recency (R)**: Number of days since the last purchase.
* **Frequency (F)**: Total number of transactions.
* **Monetary (M)**: Total amount spent by the customer.

### 2. Number of Returns

* Count of transactions where items were returned (identified via negative quantities or cancellation invoices).

### 3. Multiple Purchases Indicator

* Binary feature indicating whether a customer made more than one purchase.

### 4. Average Days Between Purchases

* Average time gap (in days) between consecutive purchases.

## 🧠 Modeling

### Algorithms Considered

* Logistic Regression
* Random Forest
* XGBoost

Among the tested models, **XGBoost** achieved the best performance and was selected as the final model.

### Model Optimization

Hyperparameter tuning was performed using **Optuna** combined with cross-validation.

The classification threshold was optimized with a focus on **recall**, as correctly identifying potentially churning customers is typically more important than overall accuracy in churn prediction scenarios.

## 📈 Evaluation

The model was evaluated using cross-validation, with particular attention to:

* Recall
* Precision
* ROC-AUC

Higher recall ensures that most at-risk customers are identified, even at the cost of some false positives.

## 🧪 Results

The model performance was evaluated across different decision thresholds. The results are summarized below:

| Thresh | Prec_1 | Rec_1 | F1_1 |
|--------|--------|-------|------|
| 0.25   | 0.668  | 0.950 | 0.784 |
| **0.30**   | **0.680**  | **0.937** | **0.788** |
| 0.35   | 0.692  | 0.900 | 0.783 |
| 0.40   | 0.708  | 0.889 | 0.788 |
| 0.45   | 0.725  | 0.820 | 0.769 |
| 0.50   | 0.746  | 0.738 | 0.742 |

###  Threshold Selection

The threshold **0.3** was chosen as the final operating point due to its strong balance between recall and precision.

At this threshold, the model achieves **very high recall (0.937)** and **moderate precision (0.680)**. This means that while some non-churners may be incorrectly flagged, the model successfully captures most potential churners.

In practice, this choice prioritizes minimizing missed churners, ensuring a low risk of failing to identify customers who are likely to churn.


## 📁 Repository Structure

### Notebooks

* **eda.ipynb**

  * Contains exploratory data analysis (EDA)
  * Feature engineering (including RFM and behavioral features)
  * Model training
  * Hyperparameter optimization using Optuna

### Scripts

* **scripts/run_pipeline.py**

  * Script to run the full churn prediction pipeline
  * Includes data loading, training, and evaluation
  * Integrated with MLflow for experiment tracking

```bash
python scripts/run_pipeline.py \
  --input data/features.csv \
  --threshold 0.30 \
  --test_size 0.2 \
  --experiment Churn \
  --mlflow_uri ./mlruns
```

### Prediction Pipeline

The repository also includes a prediction pipeline for evaluating or generating predictions using a trained model.

```bash
python scripts/run_predict.py \
  --model_root ./src/app/models \
  --experiment-name m-1ed936af9d1748ad9cef4e624c5951b2
````

## ▶️ Running the App (Streamlit + FastAPI)

This project uses:
- **FastAPI** as the backend API for predictions  
- **Streamlit** as the frontend UI  

### 1. Start FastAPI backend

From the project root directory, run:

```bash
cd src/
uvicorn app.main:app --reload --port 8000
````

- API will be available at:  
  http://localhost:8000  
- Docs (Swagger UI):  
  http://localhost:8000/docs  

  
### 2. Start Streamlit frontend

Open a **new terminal** and run:

```bash
streamlit run src/app/streamlit_app.py
````

- App will open at:  
  http://localhost:8501  


### 3. How it works

- Streamlit collects user input from the UI  
- Sends a POST request to the FastAPI endpoint:

POST http://localhost:8000/predict

- FastAPI processes the data and returns a prediction  
- Streamlit displays the result  

---
> ⚠️ **Note:**: Make sure **FastAPI is running before using Streamlit**, 
> both services must be running at the same time
---

## 🐳 Docker

### Build docker image locally

```bash
docker build -t churn-demo .
docker run -p 8000:8000 -p 8501:8501 churn-demo
````

or

### Run directly from **GitHub Container Registry**.

```bash
docker run -p 8000:8000 ghcr.io/pawlaka/churn-demo-v-1.0:latest
````

---
> ⚠️ **Note:** On first run, the Docker image may take a few minutes to start.
---

## 📝 Summary

This project demonstrates a full pipeline for churn prediction:

1. Data cleaning and preprocessing
2. Aggregation at the customer level
3. Feature engineering (RFM and behavioral metrics)
4. Model selection and optimization
5. Evaluation with business-oriented metrics

The final model (XGBoost) provides a strong baseline for identifying customers at risk of churn and can be integrated into retention strategies.
