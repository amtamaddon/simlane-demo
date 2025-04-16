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
    n_buyers     = st.number_input("Number of Buyers", 100, 5000, 500, step=100)
    urban_pct    = st.slider("% Urban Buyers", 0, 100, 60)
    high_inc_pct = st.slider("% High‑Income Buyers", 0, 100, 30)
    weeks        = st.slider("Simulation Weeks", 1, 20, 5)
    st.markdown("*Note: Adjust these parameters to match Simlane client data inputs (e.g. segment distribution, demographics).*")

with st.sidebar.expander("2) Brand Traits", expanded=False):
    def trait_sliders(label, defaults):
        return {k: st.slider(f"{label} {k}", vmin, vmax, default, step)
                for (k, (vmin, vmax, default, step)) in defaults.items()}

    simlane_traits = trait_sliders("Simlane", {
        "Price Tier":     (1, 5, 3, 1),
        "Innovation":     (0.0, 1.0, 0.8, 0.05),
        "Trust":          (0.0, 1.0, 0.75, 0.05),
        "Influencer":     (0.0, 1.0, 0.4, 0.05),
    })
    rival_traits = trait_sliders("Rival", {
        "Price Tier":     (1, 5, 2, 1),
        "Innovation":     (0.0, 1.0, 0.5, 0.05),
        "Trust":          (0.0, 1.0, 0.6, 0.05),
        "Influencer":     (0.0, 1.0, 0.7, 0.05),
    })

with st.sidebar.expander("3) Campaign Sequence", expanded=False):
    event_options = ["Price Cut", "Influencer Boost", "Bad PR"]
    campaign = st.multiselect("Sequence of weekly events", event_options, default=event_options[:1])

# ─── Buyer Class & Population ─────────────────────────────────────────────────
class Buyer:
    def __init__(self, idx, segment_profile):
        self.id = idx
        self.segment        = segment_profile["name"]
        self.price_sens     = segment_profile["price_sensitivity"]
        self.trendiness     = segment_profile["trendiness"]
        self.income         = "High" if random.random() < high_inc_pct/100 else "Low"
        self.origin         = "Urban" if random.random() < urban_pct/100 else "Rural"
        self.switch_cost    = random.uniform(0.1, 0.9)
        self.influence      = random.randint(1, 10)
        self.brand          = "Simlane" if random.random() < 0.6 else "Rival"
        self.friends        = []
        self.history        = []
        self.emotion        = "neutral"

    def update_emotion(self, new_emotion):
        self.emotion = new_emotion

# ─── Build Buyer Population ────────────────────────────────────────────────────
def build_buyers(n):
    profiles = [
        {"name":"Tech Enthusiast","weight":0.2,"price_sensitivity":0.3,"trendiness":0.8},
        {"name":"Eco-Conscious","weight":0.2,"price_sensitivity":0.4,"trendiness":0.5},
        {"name":"Loyalist","weight":0.35,"price_sensitivity":0.2,"trendiness":0.1},
        {"name":"Price Sensitive","weight":0.4,"price_sensitivity":0.9,"trendiness":0.3},
        {"name":"Trend Follower","weight":0.25,"price_sensitivity":0.4,"trendiness":0.9}
    ]
    weights = [p['weight'] for p in profiles]
    buyers = [Buyer(i, profiles[random.choices(range(len(profiles)), weights)[0]]) for i in range(n)]
    G = nx.watts_strogatz_graph(n, k=6, p=0.3)
    for i in range(n):
        buyers[i].friends = list(G.neighbors(i))
    return buyers

# ─── Utility & Event Logic ─────────────────────────────────────────────────────
def compute_utils(buyers, traits):
    price_term = np.array([b.price_sens for b in buyers]) * traits['Price Tier']
    innov_term = np.array([b.trendiness for b in buyers]) * traits['Innovation']
    infl_term  = np.array([b.influence for b in buyers]) * traits['Influencer']
    trust_term = np.array([traits['Trust']] * len(buyers)) * 0.5
    cost_term  = np.array([b.switch_cost for b in buyers])
    income_raw = np.array([1 if b.income=='High' else -1 for b in buyers])
    origin_raw = np.array([1 if b.origin=='Urban' else -1 for b in buyers])
    income_term = income_raw * traits['Trust'] * 0.1
    origin_term = origin_raw * traits['Influencer'] * 0.1
    return -price_term + innov_term + infl_term + trust_term - cost_term + income_term + origin_term


def apply_event(buyers, week, event, sim_t, riv_t):
    logs = []
    utils_sim = compute_utils(buyers, sim_t)
    utils_riv = compute_utils(buyers, riv_t)
    if event == 'Bad PR':
        sim_t['Trust'] = max(0, sim_t['Trust'] - 0.1)
    for i, b in enumerate(buyers):
        if not b.friends:
            continue
        friend_diff = [
            (utils_sim[f] - utils_riv[f]) if b.brand == 'Simlane' else (utils_riv[f] - utils_sim[f])
            for f in b.friends
        ]
        social = np.tanh(np.mean(friend_diff))
        u_sim, u_riv = utils_sim[i], utils_riv[i]
        prob = 1 / (1 + np.exp(-(abs(u_sim - u_riv) + social)))
        targ = 'Simlane' if u_sim > u_riv else 'Rival'
        if targ != b.brand and random.random() < prob:
            old = b.brand
            b.brand = targ
            b.update_emotion('hopeful' if targ == 'Simlane' else 'doubtful')
            logs.append(f"[W{week}] Buyer {b.id} switched {old}->{targ} (p={prob:.2f})")
        if event == 'Price Cut' and b.segment == 'Price Sensitive':
            pct = 0.5 * (1 - b.switch_cost)
            if ((sim_t['Price Tier'] < riv_t['Price Tier'] and b.brand == 'Rival') or
               (riv_t['Price Tier'] < sim_t['Price Tier'] and b.brand == 'Simlane')):
                if random.random() < pct:
                    tmp = b.brand
                    b.brand = 'Simlane' if tmp == 'Rival' else 'Rival'
                    logs.append(f"[W{week}] Price‑Cut nudge: {tmp}->{b.brand}")
        if event == 'Influencer Boost' and b.segment == 'Trend Follower':
            boost = 0.4 * (1 - b.switch_cost)
            better = 'Simlane' if sim_t['Influencer'] > riv_t['Influencer'] else 'Rival'
            if b.brand != better and random.random() < boost:
                old = b.brand
                b.brand = better
                logs.append(f"[W{week}] Influencer‑Boost: {old}->{better}")
    for b in buyers:
        b.history.append((week, b.brand, b.emotion))
    return logs

# ─── Run Simulation & Display ─────────────────────────────────────────────────
if st.button("Run Simulation"):
    buyers = build_buyers(n_buyers)
    timeline, logs = [], []
    for week in range(1, weeks + 1):
        evt = campaign[(week - 1) % len(campaign)] if campaign else None
        logs += apply_event(buyers, week, evt, simlane_traits, rival_traits)
        counts = pd.Series([b.brand for b in buyers]).value_counts()
        timeline.append({
            "Week": week,
            "Simlane": counts.get("Simlane", 0),
            "Rival": counts.get("Rival", 0)
        })
    df_tl = pd.DataFrame(timeline).set_index("Week")

    st.subheader("Brand Adoption Over Time")
    st.line_chart(df_tl)
    st.subheader("Final Brand Shares")
    st.table(df_tl.iloc[-1].to_frame().T)
    st.subheader("Sample Switch Logs")
    with st.expander("View Logs", expanded=False):
        for log in logs[:50]:
            st.text(log)

    # Narrative Summary
    start_sim = df_tl.iloc[0]["Simlane"]
    end_sim = df_tl.iloc[-1]["Simlane"]
    delta = end_sim - start_sim
    pct_change = delta / n_buyers * 100
    st.markdown("---")
    st.subheader("📖 Narrative Summary")
    st.markdown(
        f"Over the {weeks}-week simulation, Simlane grew from {start_sim} to {end_sim} buyers, "
        f"a net gain of {delta} ({pct_change:.1f}% of the base). Rival ended with {df_tl.iloc[-1]['Rival']} buyers.\n"
        "Key drivers: “Innovation” and “Influencer Power” among high-income urban buyers. "
        "Next steps: ensure data collection on these buyer attributes and refine model with real client data."
    )
