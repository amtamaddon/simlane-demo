import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import random
import copy

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
    event_options = ["Price Cut", "Influencer Boost", "Bad PR"]
    campaign = st.multiselect("Sequence of weekly events", event_options, default=event_options[:1])

with st.sidebar.expander("4) Optimization", expanded=False):
    run_opt = st.button("Optimize Strategy")

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
        self.brand          = "Simlane" if random.random() < 0.6 else "Rival"
        self.friends        = []
        self.history        = []
        self.emotion        = "neutral"
    def update_emotion(self, new_emotion):
        self.emotion = new_emotion

# ─── Build Population ──────────────────────────────────────────────────────────
def build_population(n):
    profiles = [
        {"name":"Tech Enthusiast","weight":0.2,"price_sensitivity":0.3,"trendiness":0.8},
        {"name":"Eco-Conscious","weight":0.2,"price_sensitivity":0.4,"trendiness":0.5},
        {"name":"Loyalist","weight":0.35,"price_sensitivity":0.2,"trendiness":0.1},
        {"name":"Price Sensitive","weight":0.4,"price_sensitivity":0.9,"trendiness":0.3},
        {"name":"Trend Follower","weight":0.25,"price_sensitivity":0.4,"trendiness":0.9}
    ]
    weights = [p['weight'] for p in profiles]
    pop = [Agent(i, profiles[random.choices(range(len(profiles)), weights)[0]]) for i in range(n)]
    G = nx.watts_strogatz_graph(n, k=6, p=0.3)
    for i in range(n): pop[i].friends = list(G.neighbors(i))
    return pop

# ─── Utility & Event Logic ─────────────────────────────────────────────────────
def compute_utils(pop, traits):
    price_term = np.array([a.price_sens for a in pop]) * traits['Price Tier']
    trend_term = np.array([a.trendiness for a in pop]) * traits['Influencer']
    trust_term = np.array([traits['Trust']] * len(pop)) * 0.5
    cost_term  = np.array([a.switch_cost for a in pop])
    return -price_term + trend_term + trust_term - cost_term


def apply_event(pop, week, event, sim_t, riv_t):
    logs = []
    utils_sim = compute_utils(pop, sim_t)
    utils_riv = compute_utils(pop, riv_t)
    if event == 'Bad PR': sim_t['Trust'] = max(0, sim_t['Trust'] - 0.1)
    for i, agent in enumerate(pop):
        if not agent.friends: continue
        friend_diff = [ (utils_sim[f]-utils_riv[f]) if agent.brand=='Simlane'
                       else (utils_riv[f]-utils_sim[f]) for f in agent.friends ]
        social = np.tanh(np.mean(friend_diff))
        u_sim, u_riv = utils_sim[i], utils_riv[i]
        prob = 1/(1+np.exp(-(abs(u_sim-u_riv)+social)))
        targ = 'Simlane' if u_sim>u_riv else 'Rival'
        if targ != agent.brand and random.random()<prob:
            old = agent.brand; agent.brand = targ
            agent.update_emotion('hopeful' if targ=='Simlane' else 'doubtful')
            logs.append(f"[W{week}] Agent {agent.id} switch {old}->{targ} p={prob:.2f}")
        # event nudges (unchanged)
        if event=='Price Cut' and agent.segment=='Price Sensitive':
            pct = 0.5*(1-agent.switch_cost)
            if ((sim_t['Price Tier']<riv_t['Price Tier'] and agent.brand=='Rival')
             or (riv_t['Price Tier']<sim_t['Price Tier'] and agent.brand=='Simlane')):
                if random.random()<pct:
                    tmp = agent.brand; agent.brand = 'Simlane' if tmp=='Rival' else 'Rival'
                    logs.append(f"[W{week}] Price‑Cut nudge: {tmp}->{agent.brand}")
        if event=='Influencer Boost' and agent.segment=='Trend Follower':
            boost = 0.4*(1-agent.switch_cost)
            better = 'Simlane' if sim_t['Influencer']>riv_t['Influencer'] else 'Rival'
            if agent.brand!=better and random.random()<boost:
                old=agent.brand; agent.brand=better
                logs.append(f"[W{week}] Influencer‑Boost: {old}->{better}")
    for a in pop: a.history.append((week, a.brand, a.emotion))
    return logs

# ─── Greedy Optimizer ───────────────────────────────────────────────────────────
def greedy_opt(pop_init, weeks, sim_t, riv_t):
    pop = copy.deepcopy(pop_init)
    seq = []
    for w in range(1, weeks+1):
        best_e, best_count, best_pop = None, -1, None
        for e in ['Price Cut','Influencer Boost','Bad PR']:
            tmp_pop = copy.deepcopy(pop)
            tmp_t = copy.deepcopy(sim_t)
            logs = apply_event(tmp_pop, w, e, tmp_t, riv_t)
            count = sum(1 for a in tmp_pop if a.brand=='Simlane')
            if count>best_count: best_count, best_e, best_pop = count, e, tmp_pop
        seq.append(best_e)
        pop = best_pop
        # no trait reset: sim_t already updated if Bad PR applied
    final_share = sum(1 for a in pop if a.brand=='Simlane')/len(pop)
    return seq, final_share

# ─── Run & Display ─────────────────────────────────────────────────────────────
if st.button("Run Simulation") or run_opt:
    population = build_population(n_agents)

    if run_opt:
        best_seq, best_share = greedy_opt(population, weeks, simlane_traits.copy(), rival_traits)
        st.subheader("🔍 Optimized Campaign Sequence")
        st.write(best_seq)
        st.write(f"Projected Final Simlane Share: {best_share:.1%}")
    else:
        timeline, logs = [], []
        for week in range(1, weeks+1):
            evt = campaign[(week-1)%len(campaign)] if campaign else None
            logs += apply_event(population, week, evt, simlane_traits, rival_traits)
            cnts = pd.Series([a.brand for a in population]).value_counts()
            timeline.append({"Week":week,
                             "Simlane":cnts.get("Simlane",0),
                             "Rival":cnts.get("Rival",0)})
        df_tl = pd.DataFrame(timeline).set_index("Week")
        st.subheader("Brand Adoption Over Time")
        st.line_chart(df_tl)
        st.subheader("Final Brand Shares")
        st.table(df_tl.iloc[-1].to_frame().T)
        st.subheader("Sample Logs")
        with st.expander("", expanded=False):
            for log in logs[:50]: st.text(log)
