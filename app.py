# app.py
import streamlit as st
import pandas as pd
from simulate import run_simulation

st.set_page_config(
    page_title="🏀 NBA Player Prop Simulator",
    page_icon="🏀",
    layout="centered",
)

st.title("🏀 NBA Player Prop Simulator")
st.markdown(
    "Simulate **1,000,000 game outcomes** using player form, head-to-head, and context "
    "to predict upcoming stat lines."
)

# User inputs
col1, col2 = st.columns(2)
with col1:
    player = st.text_input("Player Name", "LeBron James")
with col2:
    opponent = st.text_input("Opponent Team", "Boston Celtics")

sim_count = st.slider("Number of Simulations", 10000, 1000000, 100000, step=10000)

if st.button("Run Simulation"):
    with st.spinner("Running simulation..."):
        try:
            results = run_simulation(player, opponent, simulations=sim_count)
            st.success("Simulation complete ✅")
            
            df = pd.DataFrame(list(results.items()), columns=["Stat", "Predicted Value"])
            st.dataframe(df, use_container_width=True)

            st.bar_chart(df.set_index("Stat"))
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

st.caption("Powered by AI Monte Carlo simulation — built for NBA props.")
