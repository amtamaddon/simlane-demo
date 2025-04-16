import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
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
        df_upload = pd.read_csv(upload)
        st.success(f"Loaded {len(df_upload)} buyers from CSV")
    n_buyers = st.number_input("Number of Buyers (if no CSV)", 100, 5000, 500, step=100)
    weeks    = st.slider("Simulation Weeks", 1, 20, 5)

# ─── Buyer Class & Population ─────────────────────────────────────────────────
class Buyer:
    def __init__(self, idx, profile):
        self.id             = idx
        self.segment        = profile.get("segment")
        self.recency        = profile.get("recency")
        self.frequency      = profile.get("frequency")
        self.monetary       = profile.get("monetary")
        self.nps            = profile.get("nps")
        self.churn_risk     = profile.get("churn_risk")
        self.referral_count = profile.get("referral_count")
        # initial brand from data or random
        self.brand          = profile.get("brand", random.choice(["Simlane","Rival"]))

# ─── Build Buyer Population ────────────────────────────────────────────────────
def build_buyers_from_csv(df):
    buyers = []
    for _, row in df.iterrows():
        profile = row.to_dict()
        buyers.append(Buyer(row.get("id", len(buyers)), profile))
    # build social graph for reference
    G = nx.watts_strogatz_graph(len(buyers), k=min(6,len(buyers)-1), p=0.3)
    for i, b in enumerate(buyers):
        b.friends = list(G.neighbors(i))
    return buyers

def build_buyers(n):
    buyers = []
    segments = ["Tech Enthusiast","Eco-Conscious","Loyalist","Price Sensitive","Trend Follower"]
    for i in range(n):
        profile = {
            "segment": random.choice(segments),
            "recency": random.randint(1,365),
            "frequency": random.randint(1,50),
            "monetary": round(random.uniform(10,1000),2),
            "nps": random.randint(0,10),
            "churn_risk": round(random.random(),2),
            "referral_count": random.randint(0,10),
            "brand": random.choice(["Simlane","Rival"]),
        }
        buyers.append(Buyer(i, profile))
    G = nx.watts_strogatz_graph(n, k=min(6,n-1), p=0.3)
    for i, b in enumerate(buyers):
        b.friends = list(G.neighbors(i))
    return buyers

# ─── Utility Calculation ───────────────────────────────────────────────────────
def compute_utils(buyers):
    # normalization constants
    max_rec, max_freq, max_mon, max_nps, max_ref = 365, 50, 1000.0, 10, 10
    utils = []
    for b in buyers:
        rec_term   = 1 - b.recency/max_rec
        freq_term  = b.frequency/max_freq
        mon_term   = b.monetary/max_mon
        nps_term   = b.nps/max_nps
        churn_term = 1 - b.churn_risk
        ref_term   = b.referral_count/max_ref
        utils.append(rec_term + freq_term + mon_term + nps_term + churn_term + ref_term)
    return np.array(utils)

# ─── Run Simulation & Display ─────────────────────────────────────────────────
if st.button("Run Simulation"):
    # load or generate buyers
    buyers = build_buyers_from_csv(df_upload) if 'df_upload' in globals() else build_buyers(n_buyers)
    # compute utilities
    utils = compute_utils(buyers)
    # benchmark utility (e.g., mean)
    benchmark = np.mean(utils)
    # assign brand based on utility
    for i, b in enumerate(buyers):
        b.brand = "Simlane" if utils[i] >= benchmark else "Rival"
    # summary table
    df = pd.DataFrame([vars(b) for b in buyers])
    share = df['brand'].value_counts(normalize=True).mul(100).round(1)

    st.subheader("Final Brand Share (%)")
    st.table(share.to_frame('Percent'))

    st.subheader("Buyer Utility Distribution")
    util_df = pd.DataFrame({
        'Buyer ID': [b.id for b in buyers],
        'Utility' : np.round(utils, 3)
    }).set_index('Buyer ID')
    st.line_chart(util_df)

    st.subheader("Sample Buyer Data & Utility")
    sample_df = df[['id','segment','recency','frequency','monetary','nps','churn_risk','referral_count','brand']].copy()
    sample_df['utility'] = np.round(utils,3)
    st.dataframe(sample_df.sample(min(20, len(sample_df))))
