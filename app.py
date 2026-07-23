# ============================================================
# RETURNLYTICS – RETURN RISK PREDICTION & ANALYTICS APP
# Streamlit App | Ankita Daweshar
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Return Risk Predictor",
    page_icon="🛒",
    layout="wide"
)

st.markdown("""
    <style>

    /* ── Global ── */
    .main { padding: 0rem 2rem; background-color: #0e0e0e; }
    body, [data-testid="stAppViewContainer"] { background-color: #0e0e0e; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #111111;
        padding-top: 0px;
    }

    /* App name in sidebar */
    .sidebar-app-name {
        padding: 22px 18px 14px 18px;
        font-size: 1.1rem;
        font-weight: 700;
        color: white;
        letter-spacing: -0.01em;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .sidebar-divider {
        border: none;
        border-top: 1px solid #252525;
        margin: 6px 12px;
    }

    .sidebar-section-label {
        color: #555555;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 10px 18px 4px 18px;
    }

    /* Hide radio label */
    [data-testid="stSidebar"] .stRadio > label { display: none !important; }

    /* Hide radio dots */
    [data-testid="stSidebar"] input[type="radio"] { display: none !important; }
    [data-testid="stSidebar"] .stRadio > div > label > div:first-child { display: none !important; }
    [data-testid="stSidebar"] .stRadio > div { gap: 2px !important; }

    /* Nav items */
    [data-testid="stSidebar"] .stRadio > div > label {
        display: flex !important;
        align-items: center !important;
        padding: 10px 14px !important;
        margin: 1px 6px !important;
        border-radius: 8px !important;
        color: #aaaaaa !important;
        font-size: 0.9rem !important;
        cursor: pointer !important;
        border: none !important;
        background: transparent !important;
        transition: background 0.15s ease !important;
        font-weight: 400 !important;
    }
    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background: rgba(255,255,255,0.06) !important;
        color: white !important;
    }
    [data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {
        background: rgba(255,255,255,0.10) !important;
        color: white !important;
        font-weight: 600 !important;
        width: 100% !important;
        display: block !important;
        box-sizing: border-box !important;
    }

    /* Make all labels full width */
    [data-testid="stSidebar"] .stRadio > div {
        width: 100% !important;
        gap: 2px !important;
    }

    [data-testid="stSidebar"] .stRadio > div > label {
        display: flex !important;
        align-items: center !important;
        padding: 10px 14px !important;
        margin: 1px 4px !important;
        border-radius: 8px !important;
        color: #aaaaaa !important;
        font-size: 0.9rem !important;
        cursor: pointer !important;
        border: none !important;
        background: transparent !important;
        transition: background 0.15s ease !important;
        font-weight: 400 !important;
        width: calc(100% - 8px) !important;
        box-sizing: border-box !important;
    }

    /* Quick stats */
    .sidebar-stats {
        background: #181818;
        border-radius: 10px;
        padding: 12px 14px;
        margin: 6px 10px;
        border: 1px solid #252525;
    }
    .sidebar-stat-row {
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid #222222;
        font-size: 0.78rem;
    }
    .sidebar-stat-row:last-child { border-bottom: none; }
    .sidebar-stat-label { color: #777777; }
    .sidebar-stat-value { color: #e74c3c; font-weight: 600; }

    /* Sidebar footer */
    .sidebar-footer {
        padding: 16px 18px;
        border-top: 1px solid #222222;
        margin-top: 10px;
    }
    .sidebar-footer-name {
        font-size: 0.95rem;
        font-weight: 700;
        color: #dddddd;
        margin-bottom: 3px;
    }
    .sidebar-footer-sub {
        font-size: 0.75rem;
        color: #555555;
    }

    /* ── Page Header ── */
    .page-header {
        padding: 40px 0px 10px 0px;
        margin-bottom: 10px;
    }
    .page-header-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.15;
        margin: 0 0 10px 0;
        letter-spacing: -0.02em;
    }
    .page-header-subtitle {
        font-size: 1.1rem;
        color: #888888;
        margin: 0;
        font-weight: 400;
    }

    /* ── KPI Cards ── */
    .metric-card {
        background: #181818;
        border-radius: 14px;
        padding: 24px 18px;
        text-align: center;
        border: 1px solid #252525;
        border-left: 4px solid #e74c3c;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #e74c3c;
        letter-spacing: -0.02em;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #666666;
        margin-top: 6px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* ── Section subheaders ── */
    h2, h3 { color: #eeeeee !important; }

    /* ── Footer ── */
    .page-footer {
        text-align: center;
        padding: 30px 0 10px 0;
        color: #444444;
        font-size: 0.8rem;
        border-top: 1px solid #222222;
        margin-top: 30px;
    }
    .page-footer b { color: #777777; }

    </style>
""", unsafe_allow_html=True)

# ── Load model + data ──────────────────────────────────────
@st.cache_resource
def load_model():
    base     = os.path.dirname(os.path.abspath(__file__))
    model    = joblib.load(os.path.join(base, 'scripts', 'return_model.pkl'))
    scaler   = joblib.load(os.path.join(base, 'scripts', 'scaler.pkl'))
    features = json.load(open(os.path.join(base, 'scripts', 'features.json')))
    encoders = json.load(open(os.path.join(base, 'scripts', 'encoders.json')))
    return model, scaler, features, encoders

@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    return pd.read_csv(os.path.join(base, 'output', 'all_orders_with_risk_scores.csv'))

model, scaler, features, encoders = load_model()
df = load_data()

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:

    st.markdown('<div class="sidebar-app-name">🛒 &nbsp;Return Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">Menu</div>', unsafe_allow_html=True)
    page = st.radio("Navigation",
        ["🏠  Home", "📊  EDA Insights", "🔍  Risk Predictor", "📈  Model & Insights"],
        label_visibility="collapsed")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">Quick Stats</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="sidebar-stats">
            <div class="sidebar-stat-row">
                <span class="sidebar-stat-label">Total Orders</span>
                <span class="sidebar-stat-value">5,000</span>
            </div>
            <div class="sidebar-stat-row">
                <span class="sidebar-stat-label">Return Rate</span>
                <span class="sidebar-stat-value">29%</span>
            </div>
            <div class="sidebar-stat-row">
                <span class="sidebar-stat-label">High Risk Orders</span>
                <span class="sidebar-stat-value">2,406</span>
            </div>
            <div class="sidebar-stat-row">
                <span class="sidebar-stat-label">Model Accuracy</span>
                <span class="sidebar-stat-value">71%</span>
            </div>
            <div class="sidebar-stat-row">
                <span class="sidebar-stat-label">Return Cost</span>
                <span class="sidebar-stat-value">&#8377;2.9L</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown("""
        <div class="sidebar-footer">
            <div class="sidebar-footer-name">Ankita Daweshar</div>
            <div class="sidebar-footer-sub">Data Analyst &nbsp;·&nbsp; 2026</div>
        </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════
if page == "🏠  Home":

    st.markdown("""
        <div class="page-header">
            <div class="page-header-title">🧠 Returnlytics – Return Risk<br>Prediction & Analytics</div>
            <div class="page-header-subtitle">
                An end-to-end data analytics and machine learning project to identify
                and predict product return risk across 5,000 orders.
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value">5,000</div><div class="metric-label">Orders Analysed</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-value">29%</div><div class="metric-label">Return Rate</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-value">&#8377;2.9L</div><div class="metric-label">Return Cost</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-value">71%</div><div class="metric-label">Model Accuracy</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("🎯 Problem Statement")
        st.markdown("""
        An e-commerce company is losing money on product returns.
        Every returned order costs **&#8377;200** in reverse logistics.

        The business team wants to know:
        - Which products are being returned most?
        - Which cities and payment methods drive returns?
        - Can we **predict** which orders will be returned before dispatch?
        """)
    with col_r:
        st.subheader("✅ What This App Does")
        st.markdown("""
        This tool provides three things:
        - 📊 **EDA Insights** — Visual analysis of return patterns across 5,000 orders
        - 🔍 **Risk Predictor** — Enter order details, get return risk score instantly
        - 📈 **Model & Insights** — ML model performance and business recommendations

        Built with **Python · Pandas · Scikit-learn · Streamlit**
        """)

    st.divider()
    st.subheader("📌 Key Findings at a Glance")
    st.dataframe(pd.DataFrame({
        "Finding": [
            "Highest return rate category",
            "Lowest return rate category",
            "Riskiest city",
            "Riskiest payment method",
            "Top return reason",
            "High risk orders flagged by ML"
        ],
        "Value": [
            "Clothing — 37.4%",
            "Books — 22.6%",
            "City44 — 50% return rate",
            "Credit Card — 31%",
            "Defective Products",
            "2,406 out of 5,000"
        ]
    }), use_container_width=True, hide_index=True)

    st.markdown("""
        <div class="page-footer">
            <b>Ankita Daweshar</b> &nbsp;·&nbsp; E-Commerce Return Rate Reduction Analysis &nbsp;·&nbsp; 2026
        </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE 2 — EDA INSIGHTS
# ══════════════════════════════════════════════════════════
elif page == "📊  EDA Insights":

    st.markdown("""
        <div class="page-header">
            <div class="page-header-title">📊 EDA Insights</div>
            <div class="page-header-subtitle">
                Visual analysis of return patterns across 5,000 e-commerce orders
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Return Rate by Category")
        cat = df.groupby('Product_Category').apply(
            lambda x: (x['Return_Status'] == 'Returned').mean() * 100
        ).sort_values(ascending=False).reset_index()
        cat.columns = ['Category', 'Return Rate %']
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#181818')
        ax.set_facecolor('#181818')
        sns.barplot(data=cat, x='Return Rate %', y='Category', palette='Reds_r', ax=ax)
        ax.axvline(x=29, color='white', linestyle='--', linewidth=1.2, label='Avg 29%')
        ax.legend(labelcolor='white')
        ax.set_title('Return Rate by Product Category', color='white')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        for spine in ax.spines.values(): spine.set_edgecolor('#333333')
        for i, v in enumerate(cat['Return Rate %']):
            ax.text(v + 0.3, i, f'{v:.1f}%', va='center', fontweight='bold', fontsize=9, color='white')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Return Status Distribution")
        status = df['Return_Status'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#181818')
        ax.set_facecolor('#181818')
        ax.pie(status.values, labels=status.index, autopct='%1.1f%%',
               colors=['#2ecc71', '#e74c3c'], startangle=90,
               wedgeprops={'edgecolor': '#111111', 'linewidth': 2},
               textprops={'color': 'white'})
        ax.set_title('Return Status Distribution', color='white')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Return Rate by Payment Method")
        pay = df.groupby('Payment_Method').apply(
            lambda x: (x['Return_Status'] == 'Returned').mean() * 100
        ).sort_values(ascending=False).reset_index()
        pay.columns = ['Payment Method', 'Return Rate %']
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#181818'); ax.set_facecolor('#181818')
        sns.barplot(data=pay, x='Payment Method', y='Return Rate %', palette='Blues_r', ax=ax)
        ax.axhline(y=29, color='white', linestyle='--', linewidth=1.2, label='Avg 29%')
        ax.legend(labelcolor='white')
        ax.set_title('Return Rate by Payment Method', color='white')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white')
        for spine in ax.spines.values(): spine.set_edgecolor('#333333')
        for i, v in enumerate(pay['Return Rate %']):
            ax.text(i, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=9, color='white')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col4:
        st.subheader("Return Rate by Shipping Method")
        ship = df.groupby('Shipping_Method').apply(
            lambda x: (x['Return_Status'] == 'Returned').mean() * 100
        ).sort_values(ascending=False).reset_index()
        ship.columns = ['Shipping Method', 'Return Rate %']
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#181818'); ax.set_facecolor('#181818')
        sns.barplot(data=ship, x='Shipping Method', y='Return Rate %', palette='Purples_r', ax=ax)
        ax.axhline(y=29, color='white', linestyle='--', linewidth=1.2, label='Avg 29%')
        ax.legend(labelcolor='white')
        ax.set_title('Return Rate by Shipping Method', color='white')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white')
        for spine in ax.spines.values(): spine.set_edgecolor('#333333')
        for i, v in enumerate(ship['Return Rate %']):
            ax.text(i, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=9, color='white')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("---")
    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Top 10 High Risk Cities")
        city = df.groupby('User_Location').apply(
            lambda x: (x['Return_Status'] == 'Returned').mean() * 100
        ).sort_values(ascending=False).head(10).reset_index()
        city.columns = ['City', 'Return Rate %']
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor('#181818'); ax.set_facecolor('#181818')
        sns.barplot(data=city, x='Return Rate %', y='City', palette='OrRd_r', ax=ax)
        ax.axvline(x=29, color='white', linestyle='--', linewidth=1.2, label='Avg 29%')
        ax.legend(labelcolor='white')
        ax.set_title('Top 10 High Risk Cities', color='white')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white')
        for spine in ax.spines.values(): spine.set_edgecolor('#333333')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col6:
        st.subheader("Risk Label Distribution")
        risk = df['Risk_Label'].value_counts().reset_index()
        risk.columns = ['Risk Label', 'Count']
        colors_risk = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c', 'Very High': '#8e44ad'}
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor('#181818'); ax.set_facecolor('#181818')
        bars = ax.bar(risk['Risk Label'], risk['Count'],
                      color=[colors_risk.get(r, '#3498db') for r in risk['Risk Label']],
                      edgecolor='#111111')
        ax.set_title('Orders by ML Risk Label', color='white')
        ax.set_ylabel('Number of Orders', color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_edgecolor('#333333')
        for bar, val in zip(bars, risk['Count']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    str(val), ha='center', fontweight='bold', color='white')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("""<div class="page-footer">
        <b>Ankita Daweshar</b> &nbsp;·&nbsp; E-Commerce Return Rate Reduction Analysis &nbsp;·&nbsp; 2026
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE 3 — RISK PREDICTOR
# ══════════════════════════════════════════════════════════
elif page == "🔍  Risk Predictor":

    st.markdown("""
        <div class="page-header">
            <div class="page-header-title">🔍 Return Risk Predictor</div>
            <div class="page-header-subtitle">
                Enter order details below to get an instant ML-powered return risk prediction.
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        product_category = st.selectbox("Product Category",  options=encoders['Product_Category'])
        product_price    = st.number_input("Product Price (₹)", min_value=100.0, max_value=2000.0, value=800.0, step=50.0)
        discount_applied = st.slider("Discount Applied (%)", 0, 60, 20)
        order_quantity   = st.number_input("Order Quantity", 1, 5, 1)
    with col2:
        payment_method  = st.selectbox("Payment Method",  options=encoders['Payment_Method'])
        shipping_method = st.selectbox("Shipping Method", options=encoders['Shipping_Method'])
        user_age        = st.number_input("Customer Age", 18, 70, 30)
        user_gender     = st.selectbox("Customer Gender", options=encoders['User_Gender'])

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 Predict Return Risk", use_container_width=True, type="primary"):

        input_dict = {
            'Product_Category': product_category, 'Product_Price': product_price,
            'Order_Quantity'  : order_quantity,   'Discount_Applied': discount_applied,
            'Shipping_Method' : shipping_method,  'Payment_Method': payment_method,
            'User_Age'        : user_age,          'User_Gender': user_gender,
            'Order_Year'      : 2025,              'Order_Month': 6,
            'Order_Value'     : product_price * order_quantity * (1 - discount_applied / 100),
        }
        for col in ['Product_Category', 'Shipping_Method', 'Payment_Method', 'User_Gender']:
            input_dict[col] = encoders[col].index(input_dict[col])

        input_df     = pd.DataFrame([input_dict])[features]
        input_scaled = scaler.transform(input_df)
        risk_score   = model.predict_proba(input_scaled)[0][1]

        if risk_score < 0.45:
            risk_label, bg, border, txt, emoji, expl = (
                "Low Risk", "#0d2418", "#28a745", "#2ecc71", "✅",
                "This order has a low probability of return. Safe to dispatch.")
        elif risk_score < 0.52:
            risk_label, bg, border, txt, emoji, expl = (
                "Medium Risk", "#2a2000", "#ffc107", "#f39c12", "⚠️",
                "Moderate return risk. Consider verifying order details before dispatch.")
        else:
            risk_label, bg, border, txt, emoji, expl = (
                "High Risk", "#2a0a0a", "#dc3545", "#e74c3c", "🔴",
                "High return risk detected! Recommend quality check before dispatch.")

        st.divider()
        st.markdown(f"""
            <div style="background:{bg}; border-left:5px solid {border};
                        padding:28px 24px; border-radius:12px; margin-bottom:20px;">
                <div style="font-size:2.2rem; font-weight:800; color:{txt}; margin-bottom:8px;">
                    {emoji} &nbsp;{risk_label} of Return
                </div>
                <div style="font-size:1rem; color:#bbbbbb;">{expl}</div>
            </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Risk Score", f"{risk_score:.1%}")
        c2.metric("Category",    product_category)
        c3.metric("Payment",     payment_method)
        c4.metric("Discount",    f"{discount_applied}%")

        st.markdown("**🔎 Key Risk Factors:**")
        factors = []
        if discount_applied > 30:
            factors.append(f"🔸 High discount ({discount_applied}%) increases impulse buying and returns")
        if product_category == "Clothing":
            factors.append("🔸 Clothing category has the highest return rate at 37.4%")
        if payment_method in ["Credit Card", "COD"]:
            factors.append(f"🔸 {payment_method} is associated with higher return rates")
        if shipping_method == "Standard":
            factors.append("🔸 Standard shipping shows a slightly higher return rate")
        if not factors:
            factors.append("✅ No major risk factors detected for this order")
        for f in factors:
            st.markdown(f"- {f}")

    st.markdown("""<div class="page-footer">
        <b>Ankita Daweshar</b> &nbsp;·&nbsp; E-Commerce Return Rate Reduction Analysis &nbsp;·&nbsp; 2026
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE 4 — MODEL & INSIGHTS
# ══════════════════════════════════════════════════════════
elif page == "📈  Model & Insights":

    st.markdown("""
        <div class="page-header">
            <div class="page-header-title">📈 Model & Business Insights</div>
            <div class="page-header-subtitle">
                ML model performance, feature importance and business recommendations
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🤖 ML Model Details")
        st.dataframe(pd.DataFrame({
            "Property": ["Algorithm", "Train/Test Split", "Training Size",
                         "Testing Size", "Accuracy", "ROC-AUC Score", "High Risk Orders Flagged"],
            "Value":    ["Logistic Regression", "80% / 20%", "4,000 orders",
                         "1,000 orders", "71.00%", "0.57", "2,406 out of 5,000"]
        }), use_container_width=True, hide_index=True)

    with col2:
        st.subheader("📊 Feature Importance")
        coef_df = pd.DataFrame({
            'Feature': features, 'Coefficient': model.coef_[0]
        }).sort_values('Coefficient', ascending=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor('#181818'); ax.set_facecolor('#181818')
        colors_coef = ['#e74c3c' if c > 0 else '#2ecc71' for c in coef_df['Coefficient']]
        ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors_coef, edgecolor='#111111')
        ax.axvline(x=0, color='white', linewidth=1)
        ax.set_title('Feature Coefficients\n(Red = increases risk | Green = decreases risk)', color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_edgecolor('#333333')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.divider()
    st.subheader("💡 Business Recommendations")
    for title, desc in [
        ("🔴 Clothing Quality Control",
         "Clothing has the highest return rate at 37.4%. Improve size guides, product descriptions, and quality checks before dispatch."),
        ("🌍 City-Level Investigation",
         "City44 has a 50% return rate. Investigate regional logistics, supplier quality, or customer behavior patterns."),
        ("💳 Payment Method Risk",
         "Credit Card and COD orders show higher return rates. Add verification steps for high-value orders."),
        ("🏷️ Discount Strategy",
         "High discounts (>30%) correlate with more returns. Avoid heavy discounts on high-return categories like Clothing."),
        ("🤖 Predictive Flagging",
         "Use this ML model to flag high-risk orders before dispatch for quality re-checks."),
        ("📦 Supplier Audit",
         "Review suppliers with high defective return rates and set quality benchmarks."),
    ]:
        with st.expander(title):
            st.write(desc)

    st.divider()
    st.subheader("🔗 Project Repository")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Project Repo:**")
        st.markdown("[github.com/DaweAnki/Ecommerce-Return-Rate-Analysis](https://github.com/DaweAnki/Ecommerce-Return-Rate-Analysis)")
    with col_b:
        st.markdown("**Portfolio:**")
        st.markdown("[github.com/DaweAnki](https://github.com/DaweAnki)")

    st.markdown("""<div class="page-footer">
        <b>Ankita Daweshar</b> &nbsp;·&nbsp; E-Commerce Return Rate Reduction Analysis &nbsp;·&nbsp; 2026
    </div>""", unsafe_allow_html=True)
