import streamlit as st
import pandas as pd
import random
import io

st.set_page_config(page_title="Simlane Strategic Simulator", layout="centered")
st.title("🧠 Simlane Strategic Scenario Simulator")
st.subheader("Customer Switching Behavior Based on Market Events")

# === Step 0: Load or Generate Agent Data ===
st.markdown("### 📁 Load Buyer Data or Generate Synthetic Population")

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
    st.markdown("Or answer a few quick questions to generate your audience:")
    urban_pct = st.slider("% Urban Customers", 0, 100, 60)
    high_income_pct = st.slider("% High Income (>$100k)", 0, 100, 30)

# === Brand Trait Configuration ===
st.markdown("### 🔧 Brand Trait Configuration")
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
def generate_population(n=500):
    segments = ['Loyalist', 'Price Sensitive', 'Trend Follower']
    population = []
    for i in range(n):
        income = "High" if random.random() < high_income_pct / 100 else "Low"
        origin = "Urban" if random.random() < urban_pct / 100 else "Rural"
        agent = {
            "id": i,
            "segment": random.choices(segments, weights=[0.4, 0.35, 0.25])[0],
            "brand": "Simlane" if random.random() < 0.6 else "Rival",
            "influence": random.randint(0, 5),
            "income": income,
            "origin": origin,
            "switching_cost": random.uniform(0.1, 0.9)
        }
        population.append(agent)
    return pd.DataFrame(population)

# === Simulation Logic ===
def simulate_event(population_df, event, simlane_traits, rival_traits):
    logs = []
    for index, agent in population_df.iterrows():
        segment = agent['segment']
        current_brand = agent['brand']
        cost = agent['switching_cost']

        if event == "Price Cut" and segment == "Price Sensitive":
            if rival_traits["Price Tier"] < simlane_traits["Price Tier"] and current_brand == "Simlane":
                if random.random() < 0.4 * (1 - cost):
                    population_df.at[index, 'brand'] = "Rival"
                    logs.append(f"Agent {index} switched to Rival due to lower price.")
            elif simlane_traits["Price Tier"] < rival_traits["Price Tier"] and current_brand == "Rival":
                if random.random() < 0.5 * (1 - cost):
                    population_df.at[index, 'brand'] = "Simlane"
                    logs.append(f"Agent {index} switched to Simlane due to lower price.")

        elif event == "Influencer Boost" and segment == "Trend Follower":
            if simlane_traits["Influencer Power"] > rival_traits["Influencer Power"] and current_brand == "Rival":
                if random.random() < 0.4 * (1 - cost):
                    population_df.at[index, 'brand'] = "Simlane"
                    logs.append(f"Agent {index} switched to Simlane influenced by trend.")
            elif rival_traits["Influencer Power"] > simlane_traits["Influencer Power"] and current_brand == "Simlane":
                if random.random() < 0.3 * (1 - cost):
                    population_df.at[index, 'brand'] = "Rival"
                    logs.append(f"Agent {index} switched to Rival influenced by trend.")

        elif event == "Bad PR" and segment == "Loyalist":
            if current_brand == "Simlane" and simlane_traits["Trust"] < 0.7:
                if random.random() < 0.3 * (1 - cost):
                    population_df.at[index, 'brand'] = "Rival"
                    logs.append(f"Agent {index} lost trust in Simlane after PR and switched.")

    return population_df, logs

# === Scenario Selection ===
event = st.selectbox("Choose a strategic event to simulate:", ["Price Cut", "Influencer Boost", "Bad PR"])

# === Run Simulation ===
if st.button("Run Simulation"):
    pop = df_input.copy() if use_uploaded_data else generate_population()
    pre_counts = pop['brand'].value_counts().to_dict()

    updated_pop, event_logs = simulate_event(pop.copy(), event, simlane_traits, rival_traits)
    post_counts = updated_pop['brand'].value_counts().to_dict()

    delta_simlane = post_counts.get("Simlane", 0) - pre_counts.get("Simlane", 0)
    delta_rival = post_counts.get("Rival", 0) - pre_counts.get("Rival", 0)

    st.subheader(f"Results: Impact of {event}")
    df_result = pd.DataFrame([
        {"Brand": "Simlane", "Before": pre_counts.get("Simlane", 0), "After": post_counts.get("Simlane", 0), "Change": delta_simlane},
        {"Brand": "Rival", "Before": pre_counts.get("Rival", 0), "After": post_counts.get("Rival", 0), "Change": delta_rival}
    ])
    st.dataframe(df_result)
    st.bar_chart(df_result.set_index("Brand"))

    st.subheader("📝 Agent-Level Decision Log")
    for log in event_logs[:50]:
        st.text(log)
