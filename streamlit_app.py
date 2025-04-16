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
    st.markdown("#### Required CSV Columns (optional): ")
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
        segments = ["Tech Enthusiast","Eco-Conscious","Loyalist","Price Sensitive","Trend Follower"]
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
        df['brand'] = np.random.choice(['Simlane','Rival'], size=n)
    weeks = st.slider("Simulation Rounds", 1, 20, 5)

# ─── Sidebar: Utility Weights ─────────────────────────────────────────────────
with st.sidebar.expander("2) Utility Weights", expanded=False):
    recency_w   = st.slider("Recency Weight", 0.0, 2.0, 1.0, 0.05)
    frequency_w = st.slider("Frequency Weight", 0.0, 2.0, 1.0, 0.05)
    monetary_w  = st.slider("Monetary Weight", 0.0, 2.0, 1.0, 0.05)
    nps_w       = st.slider("NPS Weight", 0.0, 2.0, 1.0, 0.05)
    churn_w     = st.slider("Churn Risk Weight", 0.0, 2.0, 1.0, 0.05)
    referral_w  = st.slider("Referral Count Weight", 0.0, 2.0, 1.0, 0.05)

# ─── Prepare Graph & Normalize ─────────────────────────────────────────────────
n = len(df)
G = nx.watts_strogatz_graph(n, k=min(6,n-1), p=0.3)
neighbors = {i: list(G.neighbors(i)) for i in range(n)}
# precompute maxima for normalization
df_max = {
    'recency': 365,
    'frequency': 50,
    'monetary': df['monetary'].max(),
    'nps': 10,
    'churn_risk': 1,
    'referral_count': df['referral_count'].max()
}

# ─── Voting Loop Simulation ───────────────────────────────────────────────────
def compute_personal_score(row):
    rec   = 1 - row['recency'] / df_max['recency']
    freq  = row['frequency'] / df_max['frequency']
    mon   = row['monetary'] / df_max['monetary'] if df_max['monetary']>0 else 0
    nps   = row['nps'] / df_max['nps']
    churn = 1 - row['churn_risk'] / df_max['churn_risk']
    ref   = row['referral_count'] / df_max['referral_count'] if df_max['referral_count']>0 else 0
    return (recency_w*rec + frequency_w*freq + monetary_w*mon +
            nps_w*nps + churn_w*churn + referral_w*ref)

if st.button("Run Simulation"):
    # Initial brand list
    brands = df['brand'].tolist()
    timeline_sim = []
    timeline_riv = []

    # Run voting rounds
    for _ in range(weeks):
        scores = df.apply(compute_personal_score, axis=1)
        mean_score = scores.mean()
        new_brands = []
        for i in range(n):
            # Personal vote based on score
            p_vote = 'Simlane' if scores[i] >= mean_score else 'Rival'
            # Peer vote (majority)
            peers = neighbors[i]
            sim_cnt = sum(brands[j] == 'Simlane' for j in peers)
            peer_vote = 'Simlane' if sim_cnt >= (len(peers) - sim_cnt) else 'Rival'
            # Final vote default to personal
            new_brands.append(p_vote)
        brands = new_brands
        sim_count = brands.count('Simlane')
        timeline_sim.append(sim_count)
        timeline_riv.append(n - sim_count)

    # Build timeline DataFrame
    timeline_df = pd.DataFrame({
        'Week': range(1, weeks + 1),
        'Simlane': timeline_sim,
        'Rival': timeline_riv
    }).set_index('Week')

    # Plot results
    st.subheader("Voting Simulation Results")
    st.line_chart(timeline_df)

    # Tabular view
    st.subheader("Brand Share by Week")
    st.dataframe(timeline_df)

    # Narrative summary
    sim_initial = timeline_sim[0] / n * 100
    sim_final = timeline_sim[-1] / n * 100
    st.markdown( (
        f"**Simlane share**: {sim_initial:.1f}% → {sim_final:.1f}% over {weeks} rounds "
        f"(n={n} buyers). Rival ended at {100 - sim_final:.1f}%"
    ) )

    # Segment-level breakdown
    df['final_brand'] = brands
    st.subheader("Segment-level Outcomes")
    seg_table = (
        df.groupby(['segment', 'final_brand'])
          .size()
          .unstack(fill_value=0)
    )
    st.dataframe(seg_table)

    # Sample final assignments
    st.subheader("Sample Buyer Assignments")
    sample_cols = ['id', 'segment', 'final_brand']
    st.dataframe(df[sample_cols].sample(min(20, n)))
