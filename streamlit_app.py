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
    upload = st.file_uploader("Upload Buyer Data (CSV)", type=["csv"])
    if upload:
        df_upload = pd.read_csv(upload)
        st.success(f"Loaded {len(df_upload)} buyers from CSV")
    n_buyers     = st.number_input("Number of Buyers (if no CSV)", 100, 5000, 500, step=100)
    urban_pct    = st.slider("% Urban Buyers", 0, 100, 60)
    high_inc_pct = st.slider("% High‑Income Buyers", 0, 100, 30)
    weeks        = st.slider("Simulation Weeks", 1, 20, 5)
    st.markdown("*Note: Provide real buyer attributes via CSV to drive measurable outcomes.*")

with st.sidebar.expander("2) Brand Traits", expanded=False):
    def trait_sliders(label, defaults):
        return {k: st.slider(f"{label} {k}", vmin, vmax, default, step)
                for (k, (vmin, vmax, default, step)) in defaults.items()}

    simlane_traits = trait_sliders("Simlane", {
        "Price Tier": (1, 5, 3, 1),
        "Innovation": (0.0, 1.0, 0.8, 0.05),
        "Trust":      (0.0, 1.0, 0.75, 0.05),
        "Influencer": (0.0, 1.0, 0.4, 0.05),
    })
    rival_traits = trait_sliders("Rival", {
        "Price Tier": (1, 5, 2, 1),
        "Innovation": (0.0, 1.0, 0.5, 0.05),
        "Trust":      (0.0, 1.0, 0.6, 0.05),
        "Influencer": (0.0, 1.0, 0.7, 0.05),
    })

with st.sidebar.expander("3) Campaign Sequence", expanded=False):
    event_options = ["Price Cut", "Influencer Boost", "Bad PR"]
    campaign = st.multiselect("Sequence of weekly events", event_options, default=event_options[:1])

# ─── Buyer Class & Population ─────────────────────────────────────────────────
class Buyer:
    def __init__(self, idx, profile):
        self.id = idx
        self.segment        = profile.get("segment", profile.get("name"))
        self.price_sens     = profile["price_sensitivity"]
        self.trendiness     = profile["trendiness"]
        self.income         = profile.get("income", "High" if random.random() < high_inc_pct/100 else "Low")
        self.origin         = profile.get("origin", "Urban" if random.random() < urban_pct/100 else "Rural")
        self.switch_cost    = profile.get("switch_cost", random.uniform(0.1, 0.9))
        self.influence      = profile.get("influence", random.randint(1, 10))
        self.brand          = profile.get("brand", "Simlane" if random.random() < 0.6 else "Rival")
        self.friends        = []
        self.history        = []
        self.emotion        = "neutral"
    def update_emotion(self, new_emotion):
        self.emotion = new_emotion

# ─── Build Buyer Population ────────────────────────────────────────────────────
def build_buyers_from_csv(df):
    buyers = []
    for _, row in df.iterrows():
        profile = {
            "segment": row["segment"],
            "price_sensitivity": row["price_sensitivity"],
            "trendiness": row["trendiness"],
            "income": row["income"],
            "origin": row["origin"],
            "switch_cost": row["switch_cost"],
            "influence": row["influence"],
            "brand": row["brand"]
        }
        buyers.append(Buyer(row.get("id", len(buyers)), profile))
    # preserve CSV count
    n = len(buyers)
    G = nx.watts_strogatz_graph(n, k=min(6,n-1), p=0.3)
    for i, b in enumerate(buyers): b.friends = list(G.neighbors(i))
    return buyers

def build_buyers(n):
    # synthetic fallback: segments only
    profiles = [
        {"name":"Tech Enthusiast","price_sensitivity":0.3,"trendiness":0.8},
        {"name":"Eco-Conscious","price_sensitivity":0.4,"trendiness":0.5},
        {"name":"Loyalist","price_sensitivity":0.2,"trendiness":0.1},
        {"name":"Price Sensitive","price_sensitivity":0.9,"trendiness":0.3},
        {"name":"Trend Follower","price_sensitivity":0.4,"trendiness":0.9}
    ]
    buyers = []
    for i in range(n):
        prof = random.choice(profiles)
        buyers.append(Buyer(i, prof))
    G = nx.watts_strogatz_graph(n, k=min(6,n-1), p=0.3)
    for i, b in enumerate(buyers): b.friends = list(G.neighbors(i))
    return buyers

# ─── Utility & Event Logic ─────────────────────────────────────────────────────
def compute_utils(buyers, traits):
    arr = np.array([b.price_sens for b in buyers]) * traits['Price Tier'] * -1
    arr += np.array([b.trendiness for b in buyers]) * traits['Innovation']
    arr += np.array([b.influence for b in buyers]) * traits['Influencer']
    arr += np.array([traits['Trust']] * len(buyers)) * 0.5
    arr -= np.array([b.switch_cost for b in buyers])
    arr += np.array([1 if b.income=='High' else -1 for b in buyers]) * traits['Trust'] * 0.1
    arr += np.array([1 if b.origin=='Urban' else -1 for b in buyers]) * traits['Influencer'] * 0.1
    return arr

def apply_event(buyers, week, event, sim_t, riv_t):
    logs = []
    u_sim = compute_utils(buyers, sim_t)
    u_riv = compute_utils(buyers, riv_t)
    if event == 'Bad PR': sim_t['Trust'] = max(0, sim_t['Trust'] - 0.1)
    for i, b in enumerate(buyers):
        if not b.friends: continue
        diffs = [(u_sim[f]-u_riv[f] if b.brand=='Simlane' else u_riv[f]-u_sim[f]) for f in b.friends]
        social = np.tanh(np.mean(diffs))
        p = 1/(1+np.exp(-(abs(u_sim[i]-u_riv[i])+social)))
        target = 'Simlane' if u_sim[i]>u_riv[i] else 'Rival'
        if target!=b.brand and random.random()<p:
            old=b.brand; b.brand=target; b.update_emotion('hopeful' if target=='Simlane' else 'doubtful')
            logs.append(f"[W{week}] Buyer {b.id} switch {old}->{target} p={p:.2f}")
        if event=='Price Cut' and b.segment=='Price Sensitive':
            if random.random()<0.5*(1-b.switch_cost):
                old=b.brand; b.brand='Rival' if old=='Simlane' else 'Simlane'
                logs.append(f"[W{week}] Price‑Cut nudge: {old}->{b.brand}")
        if event=='Influencer Boost' and b.segment=='Trend Follower':
            if random.random()<0.4*(1-b.switch_cost):
                best='Simlane' if sim_t['Influencer']>riv_t['Influencer'] else 'Rival'
                if b.brand!=best:
                    old=b.brand; b.brand=best
                    logs.append(f"[W{week}] Influencer‑Boost: {old}->{best}")
    for b in buyers: b.history.append((week, b.brand, b.emotion))
    return logs

# ─── Run Simulation & Display ─────────────────────────────────────────────────
if st.button("Run Simulation"):
    buyers = build_buyers_from_csv(df_upload) if 'df_upload' in globals() else build_buyers(n_buyers)
    timeline, logs = [], []
    for week in range(1, weeks+1):
        evt = campaign[(week-1)%len(campaign)] if campaign else None
        logs += apply_event(buyers, week, evt, simlane_traits, rival_traits)
        counts = pd.Series([b.brand for b in buyers]).value_counts()
        timeline.append({"Week": week, "Simlane": counts.get("Simlane",0), "Rival": counts.get("Rival",0)})
    df_tl = pd.DataFrame(timeline)

    st.subheader("Brand Adoption Over Time")
    st.line_chart(df_tl.set_index("Week"))

    st.subheader("Final Brand Shares by Week")
    df_last = df_tl.iloc[-1:].reset_index()
    df_last.columns = ['Week','Simlane','Rival']
    st.table(df_last)

    st.subheader("Sample Switch Logs")
    with st.expander("View Logs", expanded=False):
        for log in logs[:50]: st.text(log)

    # Narrative Summary
    start, end = df_tl['Simlane'].iloc[0], df_tl['Simlane'].iloc[-1]
    delta = end - start
    pct = delta/ (len(buyers)) * 100
    st.markdown("---")
    st.subheader("📖 Narrative Summary")
    st.markdown(
        f"Simlane grew from {start} → {end} buyers over {weeks} weeks (Δ{delta}, {pct:.1f}%). Rival ended at {df_tl['Rival'].iloc[-1]}."\
        " Key drivers: Innovation & Influencer Power. Next: feed real client data, refine DIP metrics."
    )
