import streamlit as st
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import random

st.set_page_config(page_title="Simlane Diffusion Explorer", layout="wide")

st.title("Simlane Onboarding — Network Diffusion Explorer")
st.markdown(
    """
Welcome! This quick interactive shows how Simlane models the spread of influence,
risk, or demand through the networks that drive your business.  
Upload your own relationship data **OR** generate a sample network below.
"""
)

# ────────────────────────────────────────────────────────────────────────────────
# 1 ▸ BUILD OR UPLOAD THE BUSINESS NETWORK
# ────────────────────────────────────────────────────────────────────────────────

st.sidebar.header("1 ▸ Build or Upload Network")
upload_file = st.sidebar.file_uploader("Edge list CSV (source,target[,weight])", type="csv")

if upload_file is not None:
    df_edges = pd.read_csv(upload_file)
    G = nx.from_pandas_edgelist(df_edges, "source", "target", edge_attr="weight", create_using=nx.Graph())
else:
    n_nodes = st.sidebar.slider("Sample network size", 20, 200, 40, 10)
    m_edges = st.sidebar.slider("Edges per new node (Barabási–Albert model)", 1, 5, 2)
    random_seed = st.sidebar.number_input("Random seed", value=42, step=1)
    G = nx.barabasi_albert_graph(n_nodes, m_edges, seed=random_seed)

# ────────────────────────────────────────────────────────────────────────────────
# 2 ▸ CHOOSE SEED NODES & DIFFUSION PARAMETERS
# ────────────────────────────────────────────────────────────────────────────────

st.sidebar.header("2 ▸ Choose Seed Nodes & Parameters")
all_nodes = list(G.nodes)
seed_default = random.sample(all_nodes, min(3, len(all_nodes)))
seed_nodes = st.sidebar.multiselect("Early adopters / shock origin", all_nodes, default=seed_default)

a_beta = st.sidebar.slider("Activation probability (β)", 0.01, 0.5, 0.15, 0.01)

# ────────────────────────────────────────────────────────────────────────────────
# Independent Cascade implementation
# ────────────────────────────────────────────────────────────────────────────────

def independent_cascade(graph, seeds, prob):
    active, frontier, steps = set(seeds), set(seeds), [list(seeds)]
    while frontier:
        new_frontier = set()
        for u in frontier:
            for v in graph.neighbors(u):
                if v not in active and random.random() < prob:
                    new_frontier.add(v)
        if not new_frontier:
            break
        steps.append(list(new_frontier))
        active.update(new_frontier)
        frontier = new_frontier
    return active, steps

# ────────────────────────────────────────────────────────────────────────────────
# 3 ▸ RUN SIMULATION
# ────────────────────────────────────────────────────────────────────────────────

if st.sidebar.button("Run Simulation"):
    with st.spinner("Simulating diffusion…"):
        active, steps = independent_cascade(G, seed_nodes, a_beta)
        cumulative = []
        total = 0
        for s in steps:
            total += len(s)
            cumulative.append(total)
        df = pd.DataFrame({
            "Step": range(len(cumulative)),
            "New adopters": [len(s) for s in steps],
            "Cumulative adopters": cumulative
        })
        st.subheader("Diffusion Steps")
        st.dataframe(df, use_container_width=True)

        fig, ax = plt.subplots()
        ax.plot(df["Step"], df["Cumulative adopters"], marker="o")
        ax.set_xlabel("Step")
        ax.set_ylabel("Cumulative Adopters")
        ax.set_title("Diffusion Curve")
        st.pyplot(fig)

        st.success(f"{len(active)} of {G.number_of_nodes()} nodes activated.")

# ────────────────────────────────────────────────────────────────────────────────
# 4 ▸ GUIDANCE PANEL
# ────────────────────────────────────────────────────────────────────────────────

st.sidebar.header("3 ▸ What do the results mean?")
st.sidebar.info(
    """
- **Steeper curves** → faster viral reach or shock propagation.  
- **Plateaus early** → consider expanding seed set or boosting β via marketing/incentives.  
- **Sparse activation** → indicates structural silos or weak ties worth bridging.  
"""
)

st.markdown("---")
st.caption("© 2025 Simlane — Turning uncertainty into confident strategy.")
