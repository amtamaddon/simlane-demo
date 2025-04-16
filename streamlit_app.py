import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import random

# ─── App Config ───────────────────────────────────────────────────────────────
st.set_page_config("Simlane Strategic Simulator", layout="wide")
st.title("Simlane Strategic Scenario Simulator")

# ─── Sidebar: Buyer Base Settings ─────────────────────────────────────────────
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
            'brand': np.random.choice(['Simlane','Rival'], n)
        })
    weeks = st.slider("Simulation Rounds", 1, 20, 5)

# ─── Sidebar: Utility Weights ──────────────────────────────────────────────────
with st.sidebar.expander("2) Utility Weights", expanded=False):
    st.markdown("**Adjust importance of each trait for personal preference**")
    recency_w   = st.slider("Recency Weight", 0.0, 2.0, 1.0, 0.05)
    frequency_w = st.slider("Frequency Weight", 0.0, 2.0, 1.0, 0.05)
    monetary_w  = st.slider("Monetary Weight", 0.0, 2.0, 1.0, 0.05)
    nps_w       = st.slider("NPS Weight", 0.0, 2.0, 1.0, 0.05)
    churn_w     = st.slider("Churn Risk Weight", 0.0, 2.0, 1.0, 0.05)
    referral_w  = st.slider("Referral Weight", 0.0, 2.0, 1.0, 0.05)

# ─── Build Social Graph ───────────────────────────────────────────────────────
n = len(df)
G = nx.watts_strogatz_graph(n, k=min(6,n-1), p=0.3)
neighbors = {i: list(G.neighbors(i)) for i in range(n)}

# ─── Voting Loop Simulation ───────────────────────────────────────────────────
def compute_personal_score(row):
    # normalize traits
    rec = 1 - row.recency/365
    freq = row.frequency/50
    mon = row.monetary/row.monetary.max() if 'monetary' in row else 0
    nps = row.nps/10
    churn = 1 - row.churn_risk
    ref = row.referral_count/row.referral_count.max()
    # weighted sum
    return (recency_w*rec + frequency_w*freq + monetary_w*mon +
            nps_w*nps + churn_w*churn + referral_w*ref)

if st.button("Run Simulation"):
    # initialize brands
    brands = df['brand'].tolist()
    timeline = []
    for _ in range(weeks):
        new_brands = []
        personal_scores = df.apply(compute_personal_score, axis=1)
        # each agent votes based on personal + peer
        for i in range(n):
            # personal vote
            p_vote = 'Simlane' if personal_scores[i] >= personal_scores.mean() else 'Rival'
            # peer vote
            peers = neighbors[i]
            sim_count = sum(brands[j]=='Simlane' for j in peers)
            r_count = len(peers) - sim_count
            peer_vote = 'Simlane' if sim_count >= r_count else 'Rival'
            # final vote: tie-break toward personal
            vote = p_vote if p_vote==peer_vote else p_vote
            new_brands.append(vote)
        brands = new_brands
        sim_share = brands.count('Simlane')
        timeline.append(sim_share)
    # results
    df_tl = pd.DataFrame({'Week': range(1, weeks+1), 'Simlane': timeline})
    st.subheader("Voting Simulation Results")
    st.line_chart(df_tl.set_index('Week'))
    final_share = brands.count('Simlane')/n*100
    st.markdown(f"**Final Simlane Share: {final_share:.1f}%**")
    # sample final brands
    df['final_brand'] = brands
    st.subheader("Sample Final Assignments")
    st.dataframe(df[['id','segment','final_brand']].sample(min(20,n)))
