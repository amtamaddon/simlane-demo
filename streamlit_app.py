import streamlit as st
import pandas as pd
import random
import io

st.set_page_config(page_title="Simlane Strategic Simulator", layout="centered")


st.title("Simlane Strategic Scenario Simulator")

# === Step 0: Load or Generate Agent Data ===
with st.expander("📁 Load Buyer Data or Generate Synthetic Population", expanded=False):
    st.markdown("#### Required CSV Columns:")
    st.code("id,segment,brand,income,origin,switching_cost")

    st.download_button("📥 Download Sample CSV", data="""id,segment,brand,income,origin,switching_cost
0,Price Sensitive,Rival,High,Urban,0.4
1,Loyalist,Simlane,Low,Rural,0.8
2,Trend Follower,Rival,High,Urban,0.3""", file_name="sample_buyers.csv")

    upload = st.file_uploader("Upload a CSV of buyer personas (optional)", type=["csv"])

use_uploaded_data = False
if upload is not None:
    try:
        df_input = pd.read_csv(upload)
        required_cols = {"id", "segment", "brand", "income", "origin", "switching_cost"}
        if required_cols.issubset(df_input.columns):
            use_uploaded_data = True
            st.success("Valid buyer data uploaded!")
        else:
            st.error("Missing required columns. Generating synthetic agents instead.")
    except Exception as e:
        st.error(f"Error reading file: {e}. Generating synthetic agents instead.")

if not use_uploaded_data:
    st.markdown("### 🎯 Audience Variables")
st.markdown("Upload a file above--or add some input below:")
    urban_pct = st.slider("% Urban Customers", 0, 100, 60)
    high_income_pct = st.slider("% High Income (>$100k)", 0, 100, 30)
    time_steps = st.slider("🕒 Number of Simulation Rounds (Weeks)", 1, 10, 3)

# === Brand Trait Configuration ===
st.markdown("---")
with st.expander("🔧 Brand Trait Configuration", expanded=False):
    simlane_traits = {
    "Price Tier": st.slider("Simlane Price Tier (1=Low, 5=High)", 1, 5, 3),
    "Innovation": st.slider("Simlane Innovation Index (0-1)", 0.0, 1.0, 0.8),
    "Trust": st.slider("Simlane Brand Trust (0-1)", 0.0, 1.0, 0.75),
    "Influencer Power": st.slider("Simlane Influencer Power (0-1)", 0.0, 1.0, 0.4)
}

    rival_traits = {
    "Price Tier": st.slider("Rival Price Tier (1=Low, 5=High)", 1, 5, 2),
    "Innovation": st.slider("Rival Innovation Index (0-1)", 0.0, 1.0, 0.5),
    "Trust": st.slider("Rival Brand Trust (0-1)", 0.0, 1.0, 0.6),
    "Influencer Power": st.slider("Rival Influencer Power (0-1)", 0.0, 1.0, 0.7)
}

# === Create or Load Population ===
import networkx as nx

def generate_population(n=500):
    segment_profiles = [
        {"name": "Tech Enthusiast", "weight": 0.2, "price_sensitivity": 0.3, "trendiness": 0.8},
        {"name": "Eco-Conscious", "weight": 0.2, "price_sensitivity": 0.4, "trendiness": 0.5},
        {"name": "Loyalist", "weight": 0.35, "price_sensitivity": 0.2, "trendiness": 0.1},
        {"name": "Price Sensitive", "weight": 0.4, "price_sensitivity": 0.9, "trendiness": 0.3},
        {"name": "Trend Follower", "weight": 0.25, "price_sensitivity": 0.4, "trendiness": 0.9}
    ]
    segment_choices = [p["name"] for p in segment_profiles]
    weights = [p["weight"] for p in segment_profiles]

    population = []
    for i in range(n):
        seg_index = random.choices(range(len(segment_profiles)), weights=weights)[0]
        profile = segment_profiles[seg_index]
        income = "High" if random.random() < high_income_pct / 100 else "Low"
        origin = "Urban" if random.random() < urban_pct / 100 else "Rural"
        agent = {
            "id": i,
            "segment": profile["name"],
            "brand": "Simlane" if random.random() < 0.6 else "Rival",
            "influence": random.randint(0, 5),
            "income": income,
            "origin": origin,
            "switching_cost": random.uniform(0.1, 0.9),
            "price_sensitivity": profile["price_sensitivity"],
            "trendiness": profile["trendiness"]
        }
        population.append(agent)
    # Build social influence graph (simple undirected for now)
    G = nx.watts_strogatz_graph(n=n, k=6, p=0.3)  # small-world graph
    for i, agent in enumerate(population):
        agent["friends"] = list(G.neighbors(i))
        agent["memory"] = []  # record brand experiences
        agent["emotion"] = "neutral"  # can be: happy, doubtful, angry, loyal
    for agent in population:
        agent['has_switched'] = False
    return pd.DataFrame(population)

# === Simulation Logic ===
def simulate_event(population_df, event, simlane_traits, rival_traits):
    def utility(agent, brand_traits):
        return (
            -agent['price_sensitivity'] * brand_traits['Price Tier']
            + agent['trendiness'] * brand_traits['Influencer Power']
            + brand_traits['Trust'] * 0.5
        ) - agent['switching_cost']
    logs = []
    new_df = population_df.copy()
    for index, agent in population_df.iterrows():
        if 'has_switched' in agent and agent['has_switched']:
            continue
        agent_utility_simlane = utility(agent, simlane_traits)
        agent_utility_rival = utility(agent, rival_traits)
        agent_friends = agent['friends']
        peer_pressure = sum([1 for f in agent_friends if population_df.at[f, 'brand'] != agent['brand']]) / max(1, len(agent_friends))

        if agent_utility_rival > agent_utility_simlane and agent['brand'] == "Simlane":
            switch_chance = min(1.0, 0.5 + 0.5 * peer_pressure)
            if random.random() < switch_chance:
                population_df.at[index, 'brand'] = "Rival"
                population_df.at[index, 'memory'] = agent['memory'] + ["Simlane"]
                population_df.at[index, 'emotion'] = "doubtful"
                logs.append(f"Agent {index} switched to Rival based on utility + social pressure.")
                population_df.at[index, 'has_switched'] = True

        elif agent_utility_simlane > agent_utility_rival and agent['brand'] == "Rival":
            switch_chance = min(1.0, 0.5 + 0.5 * peer_pressure)
            if random.random() < switch_chance:
                population_df.at[index, 'brand'] = "Simlane"
                population_df.at[index, 'memory'] = agent['memory'] + ["Rival"]
                population_df.at[index, 'emotion'] = "hopeful"
                logs.append(f"Agent {index} switched to Simlane based on utility + social pressure.")
                population_df.at[index, 'has_switched'] = True
        segment = agent['segment']
        current_brand = agent['brand']
        cost = agent['switching_cost']

        if event == "Price Cut" and segment == "Price Sensitive":
            if rival_traits["Price Tier"] < simlane_traits["Price Tier"] and current_brand == "Simlane":
                if random.random() < 0.4 * (1 - cost):
                    new_df.at[index, 'brand'] = "Rival"
                    logs.append(f"Agent {index} switched to Rival due to lower price.")
                    new_df.at[index, 'has_switched'] = True
            elif simlane_traits["Price Tier"] < rival_traits["Price Tier"] and current_brand == "Rival":
                if random.random() < 0.5 * (1 - cost):
                    new_df.at[index, 'brand'] = "Simlane"
                    logs.append(f"Agent {index} switched to Simlane due to lower price.")
                    new_df.at[index, 'has_switched'] = True

        elif event == "Influencer Boost" and segment == "Trend Follower":
            if simlane_traits["Influencer Power"] > rival_traits["Influencer Power"] and current_brand == "Rival":
                if random.random() < 0.4 * (1 - cost):
                    new_df.at[index, 'brand'] = "Simlane"
                    logs.append(f"Agent {index} switched to Simlane influenced by trend.")
                    new_df.at[index, 'has_switched'] = True
            elif rival_traits["Influencer Power"] > simlane_traits["Influencer Power"] and current_brand == "Simlane":
                if random.random() < 0.3 * (1 - cost):
                    new_df.at[index, 'brand'] = "Rival"
                    logs.append(f"Agent {index} switched to Rival influenced by trend.")
                    new_df.at[index, 'has_switched'] = True

        elif event == "Bad PR" and segment == "Loyalist":
            if current_brand == "Simlane":
                trust_delta = 0.7 - simlane_traits["Trust"]
                if trust_delta > 0 and random.random() < 0.3 * trust_delta * (1 - cost):
                    new_df.at[index, 'brand'] = "Rival"
                    logs.append(f"Agent {index} lost trust in Simlane after PR and switched.")
                    new_df.at[index, 'has_switched'] = True

    return new_df, logs

# === Scenario Selection ===
event = st.selectbox(
    "Choose a strategic event to simulate:",
    ["Price Cut", "Influencer Boost", "Bad PR"],
    help="Simulate a single pressure event on brand loyalty. More complex campaigns coming soon."
)



# === Run Simulation ===
if st.button("Run Simulation"):
    pop = df_input.copy() if use_uploaded_data else generate_population()
    timeline = []
    logs_all = []

    for week in range(1, time_steps + 1):
        pre_counts = pop['brand'].value_counts().to_dict()
        pop, logs = simulate_event(pop, event, simlane_traits, rival_traits)
        post_counts = pop['brand'].value_counts().to_dict()

        timeline.append({
            "Week": f"Week {week}",
            "Simlane": post_counts.get("Simlane", 0),
            "Rival": post_counts.get("Rival", 0)
        })
        logs_all.extend([f"[Week {week}] {log}" for log in logs])

    df_timeline = pd.DataFrame(timeline).set_index("Week")
    st.subheader(f"📈 Brand Loyalty Over {time_steps} Weeks")
    st.line_chart(df_timeline)

    st.subheader("🧩 Summary of Behavior Insights")
    net_change = df_timeline.iloc[-1] - df_timeline.iloc[0]
    delta_df = pd.DataFrame({"Brand": net_change.index, "Net Change": net_change.values})
    delta_df = delta_df.sort_values(by="Net Change", ascending=False)
    st.dataframe(delta_df.set_index("Brand"))

    total_agents = df_timeline.iloc[-1].sum()
    switch_ids = set()
    for log in logs_all:
        if "switched" in log:
            parts = log.split("Agent ")[1]
            agent_id = parts.split(" ")[0]
            switch_ids.add(agent_id)
    switch_count = len(switch_ids)
    rate = switch_count / total_agents * 100
    winning_brand = delta_df.iloc[0]['Brand']
    st.markdown(f"**Summary:** Over {time_steps} rounds, {switch_count} agents (~{rate:.1f}%) switched brands. The winning brand was **{winning_brand}**.")


    if logs_all:
        with st.expander("View sample of agent-level switching behavior", expanded=False):
            for log in logs_all[:50]:
                st.text(log)
    else:
        st.info("No switching events recorded. Try a different scenario or adjust brand traits.")
        st.info("No switching events recorded. Try a different scenario or adjust brand traits.")
