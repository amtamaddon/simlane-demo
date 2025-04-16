import streamlit as st
import pandas as pd
import random

st.set_page_config(page_title="Simlane Strategic Simulator", layout="centered")
st.title("🧠 Simlane Strategic Scenario Simulator")
st.subheader("Customer Switching Behavior Based on Market Events")

# Create population
def generate_population(n=500):
    segments = ['Loyalist', 'Price Sensitive', 'Trend Follower']
    population = []
    for i in range(n):
        agent = {
            "id": i,
            "segment": random.choices(segments, weights=[0.4, 0.35, 0.25])[0],
            "brand": "Simlane" if random.random() < 0.6 else "Rival",
            "influence": random.randint(0, 5),
        }
        population.append(agent)
    return pd.DataFrame(population)

# Define event outcomes
def simulate_event(population_df, event):
    for index, agent in population_df.iterrows():
        if event == "Price Cut":
            if agent['segment'] == "Price Sensitive" and agent['brand'] == "Rival":
                if random.random() < 0.5:
                    population_df.at[index, 'brand'] = "Simlane"
        elif event == "Influencer Boost":
            if agent['segment'] == "Trend Follower" and agent['brand'] == "Rival":
                if random.random() < 0.4:
                    population_df.at[index, 'brand'] = "Simlane"
        elif event == "Bad PR":
            if agent['brand'] == "Simlane":
                if random.random() < 0.3:
                    population_df.at[index, 'brand'] = "Rival"
    return population_df

# User selects event
event = st.selectbox("Choose a strategic event to simulate:", ["Price Cut", "Influencer Boost", "Bad PR"])

# Run button
if st.button("Run Simulation"):
    initial_pop = generate_population()
    pre_counts = initial_pop['brand'].value_counts().to_dict()

    updated_pop = simulate_event(initial_pop.copy(), event)
    post_counts = updated_pop['brand'].value_counts().to_dict()

    # Display result
    df_result = pd.DataFrame([
        {"Brand": "Simlane", "Before": pre_counts.get("Simlane", 0), "After": post_counts.get("Simlane", 0)},
        {"Brand": "Rival", "Before": pre_counts.get("Rival", 0), "After": post_counts.get("Rival", 0)}
    ])
    st.subheader(f"Results: Impact of {event}")
    st.dataframe(df_result)

    st.bar_chart(df_result.set_index("Brand"))
