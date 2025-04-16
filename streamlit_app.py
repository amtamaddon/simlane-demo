import streamlit as st
import pandas as pd
import numpy as np
import random

# ─── App Config ───────────────────────────────────────────────────────────────
st.set_page_config("Simlane Strategic Simulator", layout="wide")
st.title("Simlane Strategic Scenario Simulator")

# ─── User Controls ─────────────────────────────────────────────────────────────
# Always-visible week slider so the dynamic effect is clear
weeks = st.slider("Simulation Rounds (Weeks)", 1, 10, 3)

with st.sidebar.expander("1) Buyer Base Settings", expanded=True):
    st.markdown("#### Required CSV Columns (optional):")
    st.code(
        "id,segment,recency,frequency,monetary,nps,churn_risk,referral_count,brand"
    )
    sample_csv = (
        "id,segment,recency,frequency,monetary,nps,churn_risk,referral_count,brand\n"
        "0,Price Sensitive,10,5,200.0,8,0.1,3,Simlane\n"
        "1,Loyalist,180,2,50.5,9,0.05,1,Rival\n"
        "2,Trend Follower,30,10,500.0,7,0.2,5,Simlane"
    )
    st.download_button(
        "📥 Download Sample CSV", data=sample_csv, file_name="sample_buyers.csv"
    )
    upload = st.file_uploader("Upload Buyer Data (CSV)", type=["csv"])
    if upload:
        df = pd.read_csv(upload)
    else:
        n = st.number_input("Number of Buyers", 100, 5000, 500, step=100)
        recency_mean = st.slider("Avg Recency (days)", 1, 365, 30)
        recency_std = st.slider("Recency Std Dev", 0, 182, 10)
        freq_mean = st.slider("Avg Frequency (# purchases)", 1, 50, 5)
        freq_std = st.slider("Frequency Std Dev", 0, 25, 2)
        mon_mean = st.slider("Avg Monetary ($)", 0.0, 10000.0, 200.0)
        mon_std = st.slider("Monetary Std Dev", 0.0, 5000.0, 100.0)
        nps_mean = st.slider("Avg NPS", 0, 10, 7)
        nps_std = st.slider("NPS Std Dev", 0, 5, 1)
        churn_mean = st.slider("Avg Churn Risk", 0.0, 1.0, 0.2)
        churn_std = st.slider("Churn Risk Std Dev", 0.0, 0.5, 0.1)
        ref_mean = st.slider("Avg Referrals", 0, 20, 2)
        ref_std = st.slider("Referral Std Dev", 0, 10, 1)
        segments = [
            "Tech Enthusiast",
            "Eco-Conscious",
            "Loyalist",
            "Price Sensitive",
            "Trend Follower",
        ]
        df = pd.DataFrame({
            'id': range(n),
            'segment': [random.choice(segments) for _ in range(n)],
            'recency': np.clip(
                np.random.normal(recency_mean, recency_std, n), 1, 365
            ).astype(int),
            'frequency': np.clip(
                np.random.normal(freq_mean, freq_std, n), 1, 50
            ).astype(int),
            'monetary': np.clip(
                np.random.normal(mon_mean, mon_std, n), 0, None
            ).round(2),
            'nps': np.clip(
                np.random.normal(nps_mean, nps_std, n), 0, 10
            ).astype(int),
            'churn_risk': np.clip(
                np.random.normal(churn_mean, churn_std, n), 0, 1
            ).round(2),
            'referral_count': np.clip(
                np.random.normal(ref_mean, ref_std, n), 0, None
            ).astype(int),
        })
        df['brand'] = np.random.choice(['Simlane', 'Rival'], size=n)

with st.sidebar.expander("2) Utility Weights", expanded=False):
    recency_w = st.slider("Recency Weight", 0.0, 2.0, 1.0)
    frequency_w = st.slider("Frequency Weight", 0.0, 2.0, 1.0)
    monetary_w = st.slider("Monetary Weight", 0.0, 2.0, 1.0)
    nps_w = st.slider("NPS Weight", 0.0, 2.0, 1.0)
    churn_w = st.slider("Churn Risk Weight", 0.0, 2.0, 1.0)
    referral_w = st.slider("Referral Count Weight", 0.0, 2.0, 1.0)

# ─── Core Scoring Function ──────────────────────────────────────────────────────
def compute_score(row, df_max):
    rec = 1 - row['recency'] / 365
    freq = row['frequency'] / 50
    mon = row['monetary'] / (df_max['monetary'] or 1)
    nps = row['nps'] / 10
    churn = 1 - row['churn_risk']
    ref = row['referral_count'] / (df_max['referral_count'] or 1)
    return (
        recency_w * rec
        + frequency_w * freq
        + monetary_w * mon
        + nps_w * nps
        + churn_w * churn
        + referral_w * ref
    )

# ─── Simulation Trigger ────────────────────────────────────────────────────────
if st.button("Run Simulation"):
    df_sim = df.copy()
    start_share = df_sim['brand'].value_counts(normalize=True)['Simlane'] * 100
    history = [start_share]

    for week in range(1, weeks + 1):
        # recency increases
        df_sim['recency'] = np.minimum(df_sim['recency'] + 7, 365)
        # churn drift
        df_sim['churn_risk'] = np.clip(
            df_sim['churn_risk'] + np.random.normal(0, 0.02, len(df_sim)),
            0,
            1,
        )
        # recompute
        df_max = {
            'monetary': df_sim['monetary'].max(),
            'referral_count': df_sim['referral_count'].max(),
        }
        df_sim['utility'] = df_sim.apply(lambda r: compute_score(r, df_max), axis=1)
        df_sim['brand'] = np.where(
            df_sim['utility'] >= df_sim['utility'].mean(),
            'Simlane',
            'Rival',
        )
        history.append(df_sim['brand'].value_counts(normalize=True)['Simlane'] * 100)

    # visualize
    chart_df = pd.DataFrame(
        {'Simlane Share': history},
        index=[f"Week 0"] + [f"Week {i}" for i in range(1, weeks + 1)],
    )
    st.subheader("Simlane Share Over Time")
    st.line_chart(chart_df)

    # final summary
    final = history[-1]
    st.subheader("📖 Summary")
    st.write(f"Started at {history[0]:.1f}%, ended at {final:.1f}% after {weeks} weeks.")
    st.markdown(
        """
- **Insight:** Recency decay & churn drift drive changes.
- **Next Steps:** Introduce targeted re-engagement for early churners.
"""
    )

    st.subheader("Sample Assignments")
    st.dataframe(
        df_sim[['id', 'segment', 'brand', 'utility']].sample(10)
    )
