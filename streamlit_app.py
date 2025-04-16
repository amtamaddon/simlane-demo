import streamlit as st
import pandas as pd
import numpy as np
import random

# ─── App Config ───────────────────────────────────────────────────────────────
st.set_page_config("Simlane Strategic Simulator", layout="wide")
st.title("Simlane Strategic Scenario Simulator")

# ─── Sidebar Inputs ──────────────────────────────────────────────────────────
with st.sidebar.expander("1) Buyer Base Settings", expanded=True):
    st.markdown("#### Required CSV Columns:")
    st.code("id,segment,recency,frequency,monetary,nps,churn_risk,referral_count,brand")
    sample_csv = (
        "id,segment,recency,frequency,monetary,nps,churn_risk,referral_count,brand\n"
        "0,Price Sensitive,10,5,200.0,8,0.1,3,Simlane\n"
        "1,Loyalist,180,2,50.5,9,0.05,1,Rival\n"
        "2,Trend Follower,30,10,500.0,7,0.2,5,Simlane"
    )
    st.download_button("📥 Download Sample CSV", data=sample_csv, file_name="sample_buyers.csv")

    upload = st.file_uploader("Upload Buyer Data (CSV)", type=["csv"])
    if upload:
        df = pd.read_csv(upload)
    else:
        n = st.number_input("Number of Buyers", 100, 5000, 500, step=100)
        recency_mean = st.slider("Avg Recency (days)", 1, 365, 30)
        recency_std  = st.slider("Recency Std Dev", 0, 182, 10)
        freq_mean    = st.slider("Avg Frequency (# purchases)", 1, 50, 5)
        freq_std     = st.slider("Frequency Std Dev", 0, 25, 2)
        mon_mean     = st.slider("Avg Monetary ($)", 0.0, 10000.0, 200.0, step=10.0)
        mon_std      = st.slider("Monetary Std Dev", 0.0, 5000.0, 100.0, step=10.0)
        nps_mean     = st.slider("Avg NPS", 0, 10, 7)
        nps_std      = st.slider("NPS Std Dev", 0, 5, 1)
        churn_mean   = st.slider("Avg Churn Risk", 0.0, 1.0, 0.2, 0.05)
        churn_std    = st.slider("Churn Risk Std Dev", 0.0, 0.5, 0.1, 0.05)
        ref_mean     = st.slider("Avg Referrals", 0, 20, 2)
        ref_std      = st.slider("Referral Std Dev", 0, 10, 1)

        segments = ["Tech Enthusiast", "Eco-Conscious", "Loyalist", "Price Sensitive", "Trend Follower"]
        df = pd.DataFrame({
            'id': range(n),
            'segment': [random.choice(segments) for _ in range(n)],
            'recency': np.clip(np.random.normal(recency_mean, recency_std, n), 1, 365).astype(int),
            'frequency': np.clip(np.random.normal(freq_mean, freq_std, n), 1, 50).astype(int),
            'monetary': np.clip(np.random.normal(mon_mean, mon_std, n), 0, None).round(2),
            'nps': np.clip(np.random.normal(nps_mean, nps_std, n), 0, 10).astype(int),
            'churn_risk': np.clip(np.random.normal(churn_mean, churn_std, n), 0, 1).round(2),
            'referral_count': np.clip(np.random.normal(ref_mean, ref_std, n), 0, None).astype(int),
        })
        df['brand'] = np.random.choice(['Simlane', 'Rival'], size=n)

weeks = st.slider("Simulation Weeks", 1, 20, 5)

# ─── Utility Weights ──────────────────────────────────────────────────────────
with st.sidebar.expander("2) Utility Weights", expanded=False):
    recency_w  = st.slider("Recency Weight", 0.0, 2.0, 1.0, 0.05)
    frequency_w= st.slider("Frequency Weight", 0.0, 2.0, 1.0, 0.05)
    monetary_w = st.slider("Monetary Weight", 0.0, 2.0, 1.0, 0.05)
    nps_w      = st.slider("NPS Weight", 0.0, 2.0, 1.0, 0.05)
    churn_w    = st.slider("Churn Risk Weight", 0.0, 2.0, 1.0, 0.05)
    referral_w = st.slider("Referral Weight", 0.0, 2.0, 1.0, 0.05)

# ─── Utility Calculation ───────────────────────────────────────────────────────
def compute_utilities(data, w):
    max_vals = {
        'recency': 365, 'frequency': 50, 'monetary': data['monetary'].max(),
        'nps': 10, 'churn_risk': 1, 'referral_count': data['referral_count'].max()
    }
    util = (
        w['recency'] * (1 - data['recency'] / max_vals['recency']) +
        w['frequency'] * (data['frequency'] / max_vals['frequency']) +
        w['monetary'] * (data['monetary'] / max_vals['monetary']) +
        w['nps'] * (data['nps'] / max_vals['nps']) +
        w['churn_risk'] * (1 - data['churn_risk'] / max_vals['churn_risk']) +
        w['referral_count'] * (data['referral_count'] / max_vals['referral_count'])
    )
    return util

# ─── Run Simulation & Display ─────────────────────────────────────────────────
if st.button("Run Simulation"):
    weights = {
        'recency': recency_w, 'frequency': frequency_w,
        'monetary': monetary_w, 'nps': nps_w,
        'churn_risk': churn_w, 'referral_count': referral_w
    }
    df['utility'] = compute_utilities(df, weights)
    benchmark = df['utility'].mean()
    df['brand'] = np.where(df['utility'] >= benchmark, 'Simlane', 'Rival')

    st.subheader("Buyer Metrics Summary")
    st.dataframe(df[['recency','frequency','monetary','nps','churn_risk','referral_count']].describe().round(2))

    st.subheader("Utility Distribution")
    st.line_chart(df.set_index('id')['utility'])

    st.subheader("Final Brand Share (%)")
    share = df['brand'].value_counts(normalize=True).mul(100).round(1)
    st.table(share.to_frame('Percent'))

    st.subheader("Sample Buyers & Utility")
    sample_cols = ['id','segment','recency','frequency','monetary','nps','churn_risk','referral_count','utility','brand']
    st.dataframe(df[sample_cols].sample(min(20, len(df))))
