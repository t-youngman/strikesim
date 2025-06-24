import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
from strikesim import StrikeSimulation
import settings

st.set_page_config(page_title="StrikeSim Dashboard", layout="wide")
st.title("⚡ StrikeSim Interactive Dashboard")
st.markdown("""
Interactively explore how union structure and policy affect strike outcomes. Adjust parameters, run the simulation, and view results instantly!
""")

# --- Sidebar: Parameter Controls ---
st.sidebar.header("Simulation Parameters")

# Worker parameters
num_workers = st.sidebar.slider("Number of workers", 10, 200, settings.num_workers)
initial_wage = st.sidebar.number_input("Initial wage", value=settings.initial_wage, min_value=1.0)
target_wage = st.sidebar.number_input("Target wage", value=settings.target_wage, min_value=1.0)
initial_savings_min, initial_savings_max = st.sidebar.slider("Initial savings range", 0.0, 5000.0, settings.initial_savings_range)
initial_morale_min, initial_morale_max = st.sidebar.slider("Initial morale range", 0.0, 1.0, settings.initial_morale_range)
initial_strike_participation_rate = st.sidebar.slider("Initial strike participation rate", 0.0, 1.0, settings.initial_strike_participation_rate)

# Financial parameters
initial_employer_balance = st.sidebar.number_input("Initial employer balance", value=settings.initial_employer_balance)
initial_strike_fund = st.sidebar.number_input("Initial strike fund", value=settings.initial_strike_fund)

# Network parameters
executive_size = st.sidebar.slider("Employer executive size", 1, 10, settings.executive_size)
department_size = st.sidebar.slider("Employer department size", 1, 20, settings.department_size)
team_size = st.sidebar.slider("Employer team size", 1, 50, settings.team_size)
bargaining_committee_size = st.sidebar.slider("Union bargaining committee size", 1, 10, settings.bargaining_committee_size)
department_density = st.sidebar.slider("Union department density", 0.0, 1.0, settings.department_density)
team_density = st.sidebar.slider("Union team density", 0.0, 1.0, settings.team_density)

# Morale and policy
morale_specification = st.sidebar.selectbox("Morale specification", ["sigmoid", "linear", "no_motivation"], index=["sigmoid", "linear", "no_motivation"].index(settings.morale_specification))
participation_threshold = st.sidebar.slider("Participation threshold", 0.0, 1.0, settings.participation_threshold)
strike_pay_rate = st.sidebar.slider("Strike pay rate (fraction of wage)", 0.0, 1.0, settings.strike_pay_rate)
dues_rate = st.sidebar.slider("Union dues rate (fraction of wage)", 0.0, 0.2, settings.dues_rate)

# Duration
duration = st.sidebar.slider("Simulation duration (days)", 10, 365, settings.duration)

# --- Run Simulation Button ---
run_sim = st.sidebar.button("Run Simulation")

# --- Prepare settings dict ---
settings_dict = dict(
    num_workers=num_workers,
    initial_wage=initial_wage,
    target_wage=target_wage,
    initial_savings_range=(initial_savings_min, initial_savings_max),
    initial_morale_range=(initial_morale_min, initial_morale_max),
    initial_strike_participation_rate=initial_strike_participation_rate,
    initial_employer_balance=initial_employer_balance,
    initial_strike_fund=initial_strike_fund,
    executive_size=executive_size,
    department_size=department_size,
    team_size=team_size,
    bargaining_committee_size=bargaining_committee_size,
    department_density=department_density,
    team_density=team_density,
    morale_specification=morale_specification,
    participation_threshold=participation_threshold,
    strike_pay_rate=strike_pay_rate,
    dues_rate=dues_rate,
    duration=duration,
    start_date=settings.start_date,
    working_days=settings.working_days,
    revenue_markup=settings.revenue_markup,
    concession_threshold=settings.concession_threshold,
    concession_policy=settings.concession_policy,
    strike_pay_policy=settings.strike_pay_policy,
)

# --- Main Panel ---
if run_sim:
    st.info("Running simulation...")
    sim = StrikeSimulation(settings_dict)
    results = sim.run_simulation()
    analysis = sim.analyze_results()

    # --- Show summary stats ---
    st.subheader("Simulation Outcome")
    col1, col2, col3 = st.columns(3)
    col1.metric("Outcome", analysis['outcome'])
    col2.metric("Final Striking Workers", analysis['final_striking_workers'])
    col3.metric("Final Working Workers", analysis['final_working_workers'])
    st.write(f"**Average morale:** {analysis['average_morale']:.3f}")
    st.write(f"**Total concessions:** ${analysis['total_concessions']:.2f}")
    st.write(f"**Total strike pay distributed:** ${analysis['total_strike_pay']:.2f}")

    # --- Time Series Plots ---
    st.subheader("Time Series Plots")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(results['striking_workers'], label='Striking', color='red')
    axes[0, 0].plot(results['working_workers'], label='Working', color='blue')
    axes[0, 0].set_title('Worker Participation Over Time')
    axes[0, 0].legend()
    axes[0, 1].plot(results['average_morale'], color='purple')
    axes[0, 1].set_title('Average Worker Morale Over Time')
    axes[1, 0].plot(results['employer_balance'], color='green')
    axes[1, 0].set_title('Employer Balance Over Time')
    axes[1, 1].plot(results['union_balance'], color='orange')
    axes[1, 1].set_title('Union Strike Fund Balance Over Time')
    plt.tight_layout()
    st.pyplot(fig)

    # --- Download Results ---
    st.subheader("Download Results")
    # Save summary CSV to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_csv:
        sim.save_summary_to_csv(tmp_csv.name)
        st.download_button("Download Summary CSV", data=open(tmp_csv.name, 'rb').read(), file_name='strikesim_summary.csv')
    # Save HDF5 to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_h5:
        sim.save_to_hdf5(tmp_h5.name)
        st.download_button("Download Full HDF5 Data", data=open(tmp_h5.name, 'rb').read(), file_name='strikesim_data.h5')

    # --- Show outcome text ---
    st.success(f"Simulation complete! Outcome: {analysis['outcome']}")

else:
    st.info("Adjust parameters in the sidebar and click 'Run Simulation' to begin.") 