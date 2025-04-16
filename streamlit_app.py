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
    st.code(
        "id,segment,recency,frequency,monetary,nps,churn_risk,referral_count,brand"
    )
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
        # synthetic fallback
        n = st.number_input("Number of Buyers", 100, 5000, 500, step=100)
        segments = ["Tech Enthusiast","Eco-Conscious","Loyalist","Price Sensitive","Trend Follower"]
        df = pd.DataFrame({
            'id': range(n),
            'segment': [random.choice(segments) for _ in range(n)],
            'recency': np.random.randint(1,366,n),
            'frequency': np.random.randint(1,51,n),
            'monetary': np.round(np.random.uniform(10,1000,n),2),
            'nps': np.random.randint(0,11,n),
            'churn_risk': np.round(np.random.random(n),2),
            'referral_count': np.random.randint(0,11,n),
        })
        df['brand'] = np.random.choice(['Simlane','Rival'], size=n)
    weeks = st.slider("Simulation Weeks", 1, 20, 5)

# ─── Dynamic Utility Weights ──────────────────────────────────────────────────
with st.sidebar.expander("2) Utility Weights", expanded=False):
    st.markdown("**Set the weight for each trait**")
    # detect numeric traits excluding id, segment, brand
    trait_cols = [c for c in df.columns if c not in ['id','segment','brand'] and pd.api.types.is_numeric_dtype(df[c])]
    weights = {}
    for col in trait_cols:
        default = 1.0
        weights[col] = st.slider(f"{col} weight", 0.0, 2.0, default, 0.05)

# ─── Utility Calculation ───────────────────────────────────────────────────────
def compute_utilities(data, weights):
    util = np.zeros(len(data))
    for col, w in weights.items():
        series = data[col].astype(float)
        norm = (series - series.min()) / (series.max() - series.min() + 1e-9)
        # invert churn-like metrics
        if 'churn' in col.lower() or 'risk' in col.lower():
            norm = 1 - norm
        util += w * norm
    return util

# ─── Run Simulation & Display ─────────────────────────────────────────────────
if st.button("Run Simulation"):
    # compute utilities
    df['utility'] = compute_utilities(df, weights)
    # assign brand based on average utility
    benchmark = df['utility'].mean()
    df['brand'] = np.where(df['utility'] >= benchmark, 'Simlane', 'Rival')

    # display metrics summary
    st.subheader("Buyer Metrics Summary")
    st.dataframe(df[trait_cols].describe().round(2))

    # utility distribution
    st.subheader("Utility Distribution")
    st.line_chart(df.set_index('id')['utility'])

    # final share
    st.subheader("Final Brand Share (%)")
    share = df['brand'].value_counts(normalize=True).mul(100).round(1)
    st.table(share.to_frame('Percent'))

    # sample output
    st.subheader("Sample Buyers & Utility")
    display_cols = ['id','segment'] + trait_cols + ['utility','brand']
    st.dataframe(df[display_cols].sample(min(20,len(df))))
