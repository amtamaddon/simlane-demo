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
    n_buyers     = st.number_input("Number of Buyers (if no CSV)", 100, 5000, 500, step=100)
    weeks        = st.slider("Simulation Weeks", 1, 20, 5)

with st.sidebar.expander("2) Buyer Metrics Weights", expanded=False):
    # Realistic metrics sliders with 'Unknown' option
    recency_w = st.slider("Recency Weight", 0.0, 1.0, 0.5, 0.05)
    rec_none  = st.checkbox("Recency Unknown", key="rec_none")
    freq_w    = st.slider("Frequency Weight", 0.0, 1.0, 0.5, 0.05)
    freq_none = st.checkbox("Frequency Unknown", key="freq_none")
    mon_w     = st.slider("Monetary Weight", 0.0, 1.0, 0.5, 0.05)
    mon_none  = st.checkbox("Monetary Unknown", key="mon_none")
    nps_w     = st.slider("NPS Weight", 0.0, 1.0, 0.5, 0.05)
    nps_none  = st.checkbox("NPS Unknown", key="nps_none")
    churn_w   = st.slider("Churn Risk Weight", 0.0, 1.0, 0.5, 0.05)
    churn_none= st.checkbox("Churn Unknown", key="churn_none")
    ref_w     = st.slider("Referral Weight", 0.0, 1.0, 0.5, 0.05)
    ref_none  = st.checkbox("Referral Unknown", key="ref_none")

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
        self.brand          = profile.get("brand", random.choice(["Simlane","Rival"]))

# ─── Build Buyer Population ────────────────────────────────────────────────────
def build_buyers_from_csv(df):
    buyers = []
    for _, row in df.iterrows():
        profile = row.to_dict()
        buyers.append(Buyer(row.get("id", len(buyers)), profile))
    G = nx.watts_strogatz_graph(len(buyers), k=min(6,len(buyers)-1), p=0.3)
    for i, b in enumerate(buyers): b.friends = list(G.neighbors(i))
    return buyers

def build_buyers(n):
    buyers = []
    for i in range(n):
        profile = {
            "segment": random.choice(["Tech Enth", "Eco-Conscious", "Loyalist", "Price Sensitive", "Trend Follower"]),
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
    for i, b in enumerate(buyers): b.friends = list(G.neighbors(i))
    return buyers

# ─── Utility Calculation ───────────────────────────────────────────────────────
def compute_utils(buyers):
    # normalize constants
    max_rec, max_freq, max_mon, max_nps, max_ref = 365, 50, 1000.0, 10, 10
    utils = []
    for b in buyers:
        rec_term = 0 if rec_none else recency_w * (1 - b.recency/max_rec)
        freq_term= 0 if freq_none else freq_w * (b.frequency/max_freq)
        mon_term = 0 if mon_none else mon_w * (b.monetary/max_mon)
        nps_term = 0 if nps_none else nps_w * (b.nps/max_nps)
        churn_term = 0 if churn_none else -churn_w * b.churn_risk
        ref_term = 0 if ref_none else ref_w * (b.referral_count/max_ref)
        utils.append(rec_term + freq_term + mon_term + nps_term + churn_term + ref_term)
    return np.array(utils)

# ─── Simulation Logic ─────────────────────────────────────────────────────────
def apply_event(buyers, utils):
    # simple brand switch based on utility difference
    logs = []
    for i, b in enumerate(buyers):
        other_util = random.choice(utils)  # placeholder for rival's utility
        if utils[i] > other_util:
            orig = b.brand; b.brand = "Simlane"; logs.append(f"Buyer {b.id} -> Simlane")
        else:
            orig = b.brand; b.brand = "Rival"; logs.append(f"Buyer {b.id} -> Rival")
    return logs

# ─── Run Simulation & Display ─────────────────────────────────────────────────
event = st.selectbox("Event:", ["None (Utility Only)", "Price Cut", "Referral Boost"])

if st.button("Run Simulation"):
    buyers = build_buyers_from_csv(df_upload) if 'df_upload' in globals() else build_buyers(n_buyers)
    utils = compute_utils(buyers)
    logs = apply_event(buyers, utils)
    
    # summarize
    df = pd.DataFrame([vars(b) for b in buyers])
    share = df['brand'].value_counts(normalize=True).mul(100).round(1)
    st.subheader("Final Brand Share (%)")
    st.table(share.to_frame('Percent'))
    
    st.subheader("Sample Logs")
    with st.expander("View Logs", expanded=False):
        for log in logs[:20]: st.text(log)
