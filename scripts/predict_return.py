# ============================================================
# E-COMMERCE RETURN RATE REDUCTION ANALYSIS
# Internship Project 2 | Elevate Labs | Ankita Daweshar
# predict_return.py — Standalone Prediction Script
# ============================================================

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

warnings.filterwarnings('ignore')

# ============================================================
# STEP 1 — LOAD DATA
# ============================================================

print("=" * 55)
print("   E-COMMERCE RETURN PREDICTION SCRIPT")
print("=" * 55)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, '..', 'data',   'returns_sustainability_dataset.csv')
OUT_PATH   = os.path.join(BASE_DIR, '..', 'output')

os.makedirs(OUT_PATH, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print(f"\n✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================================
# STEP 2 — PREPROCESSING
# ============================================================

df['Order_Date']    = pd.to_datetime(df['Order_Date'])
df['Order_Year']    = df['Order_Date'].dt.year
df['Order_Month']   = df['Order_Date'].dt.month
df['Return_Binary'] = (df['Return_Status'] == 'Returned').astype(int)

print(f"✅ Preprocessing done")
print(f"   Overall Return Rate: {df['Return_Binary'].mean()*100:.1f}%")

# ============================================================
# STEP 3 — FEATURE ENGINEERING
# ============================================================

features = [
    'Product_Category',
    'Product_Price',
    'Order_Quantity',
    'Discount_Applied',
    'Shipping_Method',
    'Payment_Method',
    'User_Age',
    'User_Gender',
    'Order_Year',
    'Order_Month',
    'Order_Value',
]

target = 'Return_Binary'

categorical_cols = [
    'Product_Category',
    'Shipping_Method',
    'Payment_Method',
    'User_Gender'
]

df_ml = df[features + [target]].copy()

le = LabelEncoder()
for col in categorical_cols:
    df_ml[col] = le.fit_transform(df_ml[col])

print(f"✅ Encoding done — {len(features)} features ready")

# ============================================================
# STEP 4 — TRAIN / TEST SPLIT
# ============================================================

X = df_ml[features]
y = df_ml[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"✅ Split done — Train: {len(X_train)} | Test: {len(X_test)}")

# ============================================================
# STEP 5 — SCALING
# ============================================================

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"✅ Scaling done")

# ============================================================
# STEP 6 — TRAIN MODEL
# ============================================================

model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    solver='lbfgs',
    class_weight='balanced'
)

model.fit(X_train_scaled, y_train)
print(f"✅ Model trained — Logistic Regression (balanced)")

# ============================================================
# STEP 7 — EVALUATE MODEL
# ============================================================

y_pred       = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n{'=' * 55}")
print(f"   MODEL EVALUATION")
print(f"{'=' * 55}")
print(f"   Accuracy  : {acc*100:.2f}%")
print(f"   ROC-AUC   : {auc:.4f}")
print(f"{'=' * 55}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Not Returned', 'Returned']))

# ============================================================
# STEP 8 — GENERATE RISK SCORES FOR ALL ORDERS
# ============================================================

X_full        = df_ml[features]
X_full_scaled = scaler.transform(X_full)

df['Return_Risk_Score'] = model.predict_proba(X_full_scaled)[:, 1]
df['Risk_Label']        = pd.cut(
    df['Return_Risk_Score'],
    bins=[0, 0.3, 0.5, 0.7, 1.0],
    labels=['Low', 'Medium', 'High', 'Very High']
)

print(f"\n✅ Risk scores generated for all {len(df)} orders")
print(f"\nRisk Label Distribution:")
print(df['Risk_Label'].value_counts())

# ============================================================
# STEP 9 — EXPORT CSVs
# ============================================================

# Full scored dataset
scored_df = df[[
    'Order_ID', 'Product_ID', 'Product_Category',
    'Product_Price', 'Discount_Applied', 'Shipping_Method',
    'Payment_Method', 'User_Location', 'Return_Status',
    'Return_Risk_Score', 'Risk_Label',
    'Return_Cost', 'Profit_Loss'
]].copy()

scored_df['Return_Risk_Score'] = scored_df['Return_Risk_Score'].round(4)

scored_path = os.path.join(OUT_PATH, 'all_orders_with_risk_scores.csv')
scored_df.to_csv(scored_path, index=False)
print(f"\n✅ Saved → output/all_orders_with_risk_scores.csv")

# High risk orders only
high_risk = (scored_df[scored_df['Return_Risk_Score'] >= 0.5]
             .sort_values('Return_Risk_Score', ascending=False)
             .reset_index(drop=True))

high_risk_path = os.path.join(OUT_PATH, 'high_risk_orders.csv')
high_risk.to_csv(high_risk_path, index=False)
print(f"✅ Saved → output/high_risk_orders.csv")

# ============================================================
# STEP 10 — FINAL SUMMARY
# ============================================================

print(f"\n{'=' * 55}")
print(f"   SCRIPT COMPLETE — FINAL SUMMARY")
print(f"{'=' * 55}")
print(f"   Total Orders Processed : {len(df)}")
print(f"   Model Accuracy         : {acc*100:.2f}%")
print(f"   ROC-AUC Score          : {auc:.4f}")
print(f"   High Risk Orders       : {len(high_risk)}")
print(f"   Output Files           : 2 CSVs saved")
print(f"{'=' * 55}")
print(f"\n🚀 Done! Check output/ folder for results.")