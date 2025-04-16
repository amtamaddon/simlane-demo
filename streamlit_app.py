import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import random

# ─── App Config ───────────────────────────────────────────────────────────────
st.set_page_config("Simlane Strategic Simulator", layout="wide")
st.title("Simlane Strategic Scenario Simulator")

# ─── Sidebar Inputs ──────────────────────────────────────────────────────────
with st.sidebar.expander("1) Population Settings", expanded=True):
    n_agents     = st.number_input("Number of agents", 100, 5000, 500, step=100)
    urban_pct    = st.slider("% Urban", 0, 100, 60)
    high_inc_pct = st.slider("% High‑Income", 0, 100, 30)
    weeks        = st.slider("Simulation Weeks", 1, 20, 5)

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
    # Let user select a sequence of events
    event_options = ["Price Cut", "Influencer Boost", "Bad PR"]
    campaign = st.multiselect("Sequence of weekly events", event_options, default=event_options[:1])

# ─── Agent Class & Population ─────────────────────────────────────────────────
class Agent:
    def __init__(self, idx, segment_profile):
        self.id = idx
        self.segment        = segment_profile["name"]
        self.price_sens     = segment_profile["price_sensitivity"]
        self.trendiness     = segment_profile["trendiness"]
        self.income         = "High" if random.random() < high_inc_pct/100 else "Low"
        self.origin         = "Urban" if random.random() < urban_pct/100 else "Rural"
        self.switch_cost    = random.uniform(0.1, 0.9)
        # start brand randomly biased to Simlane
        self.brand          = "Simlane" if random.random() < 0.6 else "Rival"
        self.friends        = []
        self.history        = []   # (week, brand, emotion)
        self.emotion        = "neutral"

    def update_emotion(self, new_emotion):
        self.emotion = new_emotion

def build_population(n):
    profiles = [
        {"name":"Tech Enthusiast","weight":0.2,"price_sensitivity":0.3,"trendiness":0.8},
        {"name":"Eco-Conscious","weight":0.2,"price_sensitivity":0.4,"trendiness":0.5},
        {"name":"Loyalist","weight":0.35,"price_sensitivity":0.2,"trendiness":0.1},
        {"name":"Price Sensitive","weight":0.4,"price_sensitivity":0.9,"trendiness":0.3},
        {"name":"Trend Follower","weight":0.25,"price_sensitivity":0.4,"trendiness":0.9}
    ]
    weights = [p["weight"] for p in profiles]
    pop = [Agent(i, profiles[random.choices(range(len(profiles)), weights)[0]]) for i in range(n)]

    # Build small‑world social graph
    G = nx.watts_strogatz_graph(n, k=6, p=0.3)
    for i in range(n):
        pop[i].friends = list(G.neighbors(i))
    return pop

# ─── Utility & Switching Logic ────────────────────────────────────────────────
def compute_utils(pop, traits):
    """Vectorized utility for all agents given brand traits dict."""
    price_term = np.array([a.price_sens for a in pop]) * traits["Price Tier"]
    trend_term = np.array([a.trendiness for a in pop]) * traits["Influencer"]
    trust_term = np.array([traits["Trust"]] * len(pop)) * 0.5
    cost_term  = np.array([a.switch_cost for a in pop])
    return -price_term + trend_term + trust_term - cost_term

def apply_event(pop, week, event, sim_t, riv_t):
    """Run one week’s switches under a given event."""
    logs = []
    utils_sim = compute_utils(pop, sim_t)
    utils_riv = compute_utils(pop, riv_t)

    # dynamic trust erosion on Bad PR
    if event == "Bad PR":
        sim_t["Trust"] = max(0, sim_t["Trust"] - 0.1)

    for i, agent in enumerate(pop):
        # social influence = avg util difference among friends
        if not agent.friends: continue
        friend_utils_diff = [
            (utils_sim[f] - utils_riv[f]) if agent.brand=="Simlane"
            else (utils_riv[f] - utils_sim[f])
            for f in agent.friends
        ]
        social_pressure = np.tanh(np.mean(friend_utils_diff))

        # decide utility-driven switch
        u_sim, u_riv = utils_sim[i], utils_riv[i]
        flip_prob = 1/(1+np.exp(-(abs(u_sim-u_riv) + social_pressure)))
        target = "Simlane" if u_sim>u_riv else "Rival"
        if target != agent.brand and random.random() < flip_prob:
            old = agent.brand
            agent.brand = target
            agent.update_emotion("hopeful" if target=="Simlane" else "doubtful")
            logs.append(f"[W{week}] Agent {agent.id} switched {old}→{target} (prob={flip_prob:.2f})")

        # event‑specific nudges
        if event=="Price Cut" and agent.segment=="Price Sensitive":
            pct = 0.5*(1-agent.switch_cost)
            if ((sim_t["Price Tier"]<riv_t["Price Tier"] and agent.brand=="Rival")
             or (riv_t["Price Tier"]<sim_t["Price Tier"] and agent.brand=="Simlane")):
                if random.random()<pct:
                    tmp = agent.brand
                    agent.brand = "Simlane" if tmp=="Rival" else "Rival"
                    logs.append(f"[W{week}] Price‑Cut nudge: {tmp}→{agent.brand}")

        if event=="Influencer Boost" and agent.segment=="Trend Follower":
            boost = 0.4*(1-agent.switch_cost)
            better = ("Simlane" if sim_t["Influencer"]>riv_t["Influencer"] else "Rival")
            if agent.brand!=better and random.random()<boost:
                old = agent.brand
                agent.brand = better
                logs.append(f"[W{week}] Influencer‑Boost: {old}→{better}")

    # record history
    for agent in pop:
        agent.history.append((week, agent.brand, agent.emotion))
    return logs

# ─── Run Simulation & Display ─────────────────────────────────────────────────
if st.button("Run Simulation"):
    population = build_population(n_agents)
    timeline = []
    all_logs = []

    for week in range(1, weeks+1):
        event = campaign[(week-1) % len(campaign)] if campaign else None
        logs = apply_event(population, week, event, simlane_traits, rival_traits)
        all_logs += logs

        counts = pd.Series([a.brand for a in population]).value_counts()
        timeline.append({"Week":week, "Simlane":counts.get("Simlane",0), "Rival":counts.get("Rival",0)})

    df_tl = pd.DataFrame(timeline).set_index("Week")
    st.subheader("Brand Adoption Over Time")
    st.line_chart(df_tl)

    st.subheader("Switch Logs (first 50 entries)")
    with st.expander("", expanded=False):
        for log in all_logs[:50]:
            st.text(log)

    st.subheader("Final Brand Shares")
    st.table(df_tl.iloc[-1].to_frame().T)
