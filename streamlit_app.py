import streamlit as st
import pandas as pd
import random
import io

st.set_page_config(page_title="Simlane Strategic Simulator", layout="centered")
st.title("🧠 Simlane Strategic Scenario Simulator")
st.subheader("Customer Switching Behavior Based on Market Events")

# === Step 0: Load or Generate Agent Data ===
st.markdown("### 📁 Load Buyer Data or Generate Synthetic Population")
st.markdown("#### Required CSV Columns:")
st.code("id,segment,brand,income,origin,switching_cost")

st.download_button("📥 Download Sample CSV", data="""id,segment,brand,income,origin,switching_cost\n0,Price Sensitive,Rival,High,Urban,0.4\n1,Loyalist,Simlane,Low,Rural,0.8\n2,Trend Follower,Rival,High,Urban,0.3""", file_name="sample_buyers.csv")

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
    time_steps = st.slider("🕒 Number of Simulation Rounds (Weeks)", 1, 10, 3)

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
    segment_profiles = [
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
    return pd.DataFrame(population)

# === Simulation Logic ===
def simulate_event(population_df, event, simlane_traits, rival_traits):
    logs = []
    new_df = population_df.copy()
    for index, agent in population_df.iterrows():
        segment = agent['segment']
        current_brand = agent['brand']
        cost = agent['switching_cost']

        if event == "Price Cut" and segment == "Price Sensitive":
            if rival_traits["Price Tier"] < simlane_traits["Price Tier"] and current_brand == "Simlane":
                if random.random() < 0.4 * (1 - cost):
                    new_df.at[index, 'brand'] = "Rival"
                    logs.append(f"Agent {index} switched to Rival due to lower price.")
            elif simlane_traits["Price Tier"] < rival_traits["Price Tier"] and current_brand == "Rival":
                if random.random() < 0.5 * (1 - cost):
                    new_df.at[index, 'brand'] = "Simlane"
                    logs.append(f"Agent {index} switched to Simlane due to lower price.")

        elif event == "Influencer Boost" and segment == "Trend Follower":
            if simlane_traits["Influencer Power"] > rival_traits["Influencer Power"] and current_brand == "Rival":
                if random.random() < 0.4 * (1 - cost):
                    new_df.at[index, 'brand'] = "Simlane"
                    logs.append(f"Agent {index} switched to Simlane influenced by trend.")
            elif rival_traits["Influencer Power"] > simlane_traits["Influencer Power"] and current_brand == "Simlane":
                if random.random() < 0.3 * (1 - cost):
                    new_df.at[index, 'brand'] = "Rival"
                    logs.append(f"Agent {index} switched to Rival influenced by trend.")

        elif event == "Bad PR" and segment == "Loyalist":
            if current_brand == "Simlane":
                trust_delta = 0.7 - simlane_traits["Trust"]
                if trust_delta > 0 and random.random() < 0.3 * trust_delta * (1 - cost):
                    new_df.at[index, 'brand'] = "Rival"
                    logs.append(f"Agent {index} lost trust in Simlane after PR and switched.")

    return new_df, logs

# === Scenario Selection ===
event = st.selectbox("Choose a strategic event to simulate:", ["Price Cut", "Influencer Boost", "Bad PR"])

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

    st.subheader("📝 Agent-Level Decision Log")
    if logs_all:
        for log in logs_all[:100]:
            st.text(log)
    else:
        st.info("No switching events recorded. Try a different scenario or adjust brand traits.")
