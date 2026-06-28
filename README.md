# 📊 E-Commerce Return Rate Reduction Analysis

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Power BI](https://img.shields.io/badge/PowerBI-Dashboard-yellow?logo=powerbi)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

> A complete end-to-end E-Commerce Analytics project to understand, visualize, and predict product return rates using Exploratory Data Analysis, Machine Learning, and Power BI.

---

## 📌 Problem Statement

An e-commerce company has **5,000 orders** across 5 product categories. Every returned order costs the company ₹200 in reverse logistics — and the business doesn't know which orders will be returned or why.

**Goal →** Use data science to find **why** customers are returning products, **where** returns are happening most, and **predict** which future orders are likely to be returned.

---

## 📁 Project Structure
```Ecommerce-Return-Rate-Analysis/

│
├── data/
│   └── returns_sustainability_dataset.csv
│
├── notebooks/
│   ├── 01_EDA_Return_Analysis.ipynb
│   └── 02_ML_LogisticRegression.ipynb
│
├── scripts/
│   └── predict_return.py
│
├── output/
│   ├── plot1_overall_return_count.png
│   ├── plot2_return_rate_pie.png
│   ├── plot3_return_by_category.png
│   ├── plot4_return_reasons.png
│   ├── plot5_return_by_payment.png
│   ├── plot6_return_by_shipping.png
│   ├── plot7_high_risk_cities.png
│   ├── plot8_discount_vs_return.png
│   ├── plot9_profit_impact.png
│   ├── plot10_return_by_gender.png
│   ├── plot11_yearly_trend.png
│   ├── ml_plot1_confusion_matrix.png
│   ├── ml_plot2_roc_curve.png
│   ├── ml_plot3_feature_importance.png
│   ├── ml_plot4_risk_score_distribution.png
│   ├── high_risk_products.csv
│   ├── high_risk_orders.csv
│   └── all_orders_with_risk_scores.csv
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | Kaggle — Returns Sustainability Dataset |
| Rows | 5,000 orders |
| Columns | 23 features |
| Target | Return_Status (Returned / Not Returned) |
| Missing Values | None |
| Overall Return Rate | 29.0% |
| Date Range | January 2022 – September 2025 |

---

## 🔍 Key Findings

| # | Finding |
|---|---|
| 1 | **Clothing** has the highest return rate at **37.4%** — nearly 1 in 3 orders returned |
| 2 | **City44** is the riskiest location with a **50% return rate** |
| 3 | **Defective products** are the top return reason (26.3% of all returns) |
| 4 | **Credit Card** payments have the highest return rate at **31%** |
| 5 | Orders with **higher discounts** (avg 26.8%) are returned more than non-returned orders (avg 24.3%) |
| 6 | Total direct return cost is **₹2,90,000** across 1,450 returned orders |

---

## 🤖 Machine Learning Model

| Property | Value |
|---|---|
| Algorithm | Logistic Regression |
| Train/Test Split | 80% / 20% |
| Training Size | 4,000 orders |
| Testing Size | 1,000 orders |
| Accuracy | **71.00%** |
| ROC-AUC Score | **0.56+** |
| Class Weighting | Balanced |

---


---

## 🛠️ Tools & Technologies

| Tool | Purpose |
|---|---|
| Python | Data analysis and ML |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | EDA visualizations |
| Scikit-learn | Logistic Regression model |
| Power BI Desktop | Interactive dashboard |
| DAX | Custom KPI measures |
| GitHub + VS Code | Version control |

---

## 📦 Deliverables

- ✅ `01_EDA_Return_Analysis.ipynb` — Exploratory Data Analysis with 11 charts
- ✅ `02_ML_LogisticRegression.ipynb` — Logistic Regression Model (71% accuracy)
- ✅ `predict_return.py` — Standalone Python prediction script
- ✅ `high_risk_products.csv` — Products with ≥40% return rate
- ✅ `high_risk_orders.csv` — Orders with risk score ≥ 0.5
- ✅ `all_orders_with_risk_scores.csv` — All 5000 orders with predicted risk scores
- 
---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/DaweAnki/Ecommerce-Return-Rate-Analysis.git
cd Ecommerce-Return-Rate-Analysis
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run EDA notebook**
```bash
cd notebooks
jupyter notebook 01_EDA_Return_Analysis.ipynb
```

**4. Run ML Model notebook**
```bash
jupyter notebook 02_ML_LogisticRegression.ipynb
```

**5. Run standalone prediction script**
```bash
cd scripts
python predict_return.py
```
