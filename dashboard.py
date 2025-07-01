import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
from strikesim import StrikeSimulation
import settings
import networkx as nx
from datetime import datetime, timedelta

# --- Helper: List available network files (.gexf and .csv) ---
def list_network_files(folder):
    files = []
    if os.path.exists(folder):
        for f in os.listdir(folder):
            if f.endswith('.gexf') or f.endswith('.csv'):
                files.append(os.path.splitext(f)[0])
    return files

# --- Initialize session state ---
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'simulation_object' not in st.session_state:
    st.session_state.simulation_object = None

st.set_page_config(page_title="StrikeSim Dashboard", layout="wide")
st.title("⚡⚡⚡    StrikeSim    ⚡⚡⚡")

# --- Sidebar: Parameter Controls ---
st.sidebar.header("Simulation Parameters")

# --- Network Settings (Accordion) ---
with st.sidebar.expander("🌐 Workplace Networks", expanded=False):
    # Network selection
    employer_network_options = ['random'] + list_network_files('networks/employers')
    union_network_options = ['random'] + list_network_files('networks/unions')

    employer_network_choice = st.selectbox("Employer network", employer_network_options, index=0)
    union_network_choice = st.selectbox("Union network", union_network_options, index=0)

    # Employer network parameters (show only if random)
    if employer_network_choice == 'random':
        num_workers = st.slider("Number of workers", 10, 200, settings.num_workers)
        executive_size = st.slider("Employer executive size", 1, 10, settings.executive_size)
        department_size = st.slider("Employer department size", 1, 20, settings.department_size)
        team_size = st.slider("Employer team size", 1, 50, settings.team_size)
    else:
        num_workers = None
        executive_size = None
        department_size = None
        team_size = None

    # Union network parameters (show only if random)
    if union_network_choice == 'random':
        bargaining_committee_size = st.slider("Union bargaining committee size", 1, 10, settings.bargaining_committee_size)
        steward_percentage = st.slider("Union steward percentage", 0.01, 0.5, settings.steward_percentage, help="Fraction of workers who are union stewards (well-connected leaders)")
        team_density = st.slider("Union team density", 0.0, 1.0, settings.team_density)
    else:
        bargaining_committee_size = None
        steward_percentage = None
        team_density = None

# --- Worker Settings (Accordion) ---
with st.sidebar.expander("👥 Worker Settings", expanded=True):
    # Worker parameters
    initial_wage = st.number_input("Initial wage", value=settings.initial_wage, min_value=1.0)
    target_wage = st.number_input("Target wage", value=settings.target_wage, min_value=1.0)
    initial_savings_min, initial_savings_max = st.slider("Initial savings range", 0.0, 5000.0, settings.initial_savings_range)
    initial_morale_min, initial_morale_max = st.slider("Initial morale range", 0.0, 1.0, settings.initial_morale_range)

# --- Employer Settings (Accordion) ---
with st.sidebar.expander("🏢 Employer Settings", expanded=False):
    initial_employer_balance = st.number_input("Initial employer balance", value=settings.initial_employer_balance)
    revenue_markup = st.number_input("Revenue markup", value=settings.revenue_markup, min_value=1.0, step=0.1)
    concession_threshold = st.number_input("Concession threshold", value=settings.concession_threshold)

# --- Union Settings (Accordion) ---
with st.sidebar.expander("🤝 Union Settings", expanded=False):
    initial_strike_fund = st.number_input("Initial strike fund", value=settings.initial_strike_fund)
    strike_pay_rate = st.slider("Strike pay rate (fraction of wage)", 0.0, 1.0, settings.strike_pay_rate)
    dues_rate = st.slider("Union dues rate (fraction of wage)", 0.0, 0.2, settings.dues_rate)

# --- Simulation Settings ---
with st.sidebar.expander("⚙️ Simulation Settings", expanded=False):
    morale_specification = st.selectbox("Morale specification", ["sigmoid", "linear", "no_motivation"], index=["sigmoid", "linear", "no_motivation"].index(settings.morale_specification))
    duration = st.slider("Simulation duration (days)", 10, 365, settings.duration)

# --- Calendar Settings (Accordion) ---
with st.sidebar.expander("📅 Calendar & Strike Patterns", expanded=False):
    # Start date picker
    start_date = st.date_input(
        "Start Date",
        value=datetime.strptime(settings.start_date, '%Y-%m-%d').date(),
        help="Select the start date for the simulation"
    )
    
    # Strike pattern selection
    strike_pattern = st.selectbox(
        "Strike Pattern",
        ["indefinite", "once_a_week", "once_per_month", "weekly_escalation"],
        index=["indefinite", "once_a_week", "once_per_month", "weekly_escalation"].index(settings.strike_pattern),
        help="Choose how often strikes occur"
    )
    
    # Weekly escalation setting (only show if selected)
    if strike_pattern == "weekly_escalation":
        weekly_escalation_start = st.slider(
            "Starting Days per Week",
            min_value=1,
            max_value=5,
            value=settings.weekly_escalation_start,
            help="Number of strike days in the first week (escalates by 1 each week)"
        )
    else:
        weekly_escalation_start = settings.weekly_escalation_start
    
    # Working days selection
    st.markdown("**Working Days:**")
    col1, col2 = st.columns(2)
    with col1:
        monday = st.checkbox("Monday", value="Monday" in settings.working_days)
        tuesday = st.checkbox("Tuesday", value="Tuesday" in settings.working_days)
        wednesday = st.checkbox("Wednesday", value="Wednesday" in settings.working_days)
        thursday = st.checkbox("Thursday", value="Thursday" in settings.working_days)
    with col2:
        friday = st.checkbox("Friday", value="Friday" in settings.working_days)
        saturday = st.checkbox("Saturday", value="Saturday" in settings.working_days)
        sunday = st.checkbox("Sunday", value="Sunday" in settings.working_days)
    
    # Build working days set
    working_days = set()
    if monday: working_days.add("Monday")
    if tuesday: working_days.add("Tuesday")
    if wednesday: working_days.add("Wednesday")
    if thursday: working_days.add("Thursday")
    if friday: working_days.add("Friday")
    if saturday: working_days.add("Saturday")
    if sunday: working_days.add("Sunday")
    
    # Ensure at least one working day is selected
    if not working_days:
        st.warning("Please select at least one working day!")
        working_days = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday"}

# --- Run Simulation Button ---
run_sim = st.sidebar.button("🚀 Run Simulation", type="primary")

# --- Prepare settings dict ---
settings_dict = dict(
    num_workers=num_workers,
    initial_wage=initial_wage,
    target_wage=target_wage,
    initial_savings_range=(initial_savings_min, initial_savings_max),
    initial_morale_range=(initial_morale_min, initial_morale_max),
    initial_employer_balance=initial_employer_balance,
    initial_strike_fund=initial_strike_fund,
    morale_specification=morale_specification,
    strike_pay_rate=strike_pay_rate,
    dues_rate=dues_rate,
    duration=duration,
    start_date=start_date.strftime('%Y-%m-%d'),
    working_days=working_days,
    revenue_markup=revenue_markup,
    concession_threshold=concession_threshold,
    concession_policy=settings.concession_policy,
    strike_pay_policy=settings.strike_pay_policy,
    # Strike pattern settings
    strike_pattern=strike_pattern,
    weekly_escalation_start=weekly_escalation_start,
)

# Set network file or parameters in settings_dict
if employer_network_choice == 'random':
    settings_dict['executive_size'] = executive_size
    settings_dict['department_size'] = department_size
    settings_dict['team_size'] = team_size
    settings_dict['employer_network_file'] = None
else:
    settings_dict['employer_network_file'] = employer_network_choice

if union_network_choice == 'random':
    settings_dict['bargaining_committee_size'] = bargaining_committee_size
    settings_dict['steward_percentage'] = steward_percentage
    settings_dict['team_density'] = team_density
    settings_dict['union_network_file'] = None
else:
    settings_dict['union_network_file'] = union_network_choice

# --- Main Panel ---
if run_sim:
    st.info("Running simulation...")
    
    # Run simulation and store in session state
    sim = StrikeSimulation(settings_dict)
    results = sim.run_simulation()
    analysis = sim.analyze_results()
    
    # Store results in session state
    st.session_state.simulation_results = results
    st.session_state.simulation_object = sim

# --- Display results if available ---
if st.session_state.simulation_results is not None:
    results = st.session_state.simulation_results
    sim = st.session_state.simulation_object
    analysis = sim.analyze_results()

    # --- Show summary stats ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Simultion Outcome", analysis['outcome'])
    col2.metric("Final Striking Workers", analysis['final_striking_workers'])
    col3.metric("Final Working Workers", analysis['final_working_workers'])
    st.write(f"**Average morale:** {analysis['average_morale']:.3f}")
    st.write(f"**Average savings:** ${analysis['average_savings']:.2f}")
    st.write(f"**Total strike pay distributed:** ${analysis['total_strike_pay']:.2f}")

    # --- Time Series Plots ---
    st.subheader("Time Series Plots")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Create day numbers for x-axis
    days = list(range(len(results['striking_workers'])))
    
    # Calculate strike days for annotation
    strike_days = []
    if strike_pattern != 'indefinite':
        current_date = start_date
        for i, day in enumerate(days):
            if sim.is_strike_day(current_date):
                strike_days.append(i)
            current_date += timedelta(days=1)
    
    axes[0, 0].plot(days, results['striking_workers'], label='Striking', color='red')
    axes[0, 0].plot(days, results['working_workers'], label='Working', color='blue')
    # Add vertical lines for strike days
    for strike_day in strike_days:
        axes[0, 0].axvline(x=strike_day, color='orange', alpha=0.3, linestyle='--')
    axes[0, 0].set_title('Worker Participation Over Time')
    axes[0, 0].set_xlabel('Day of Simulation')
    axes[0, 0].legend()
    axes[0, 1].plot(days, results['average_morale'], color='purple')
    axes[0, 1].set_title('Average Worker Morale Over Time')
    axes[0, 1].set_xlabel('Day of Simulation')
    axes[0, 2].plot(days, results['average_savings'], color='brown')
    axes[0, 2].set_title('Average Worker Savings Over Time')
    axes[0, 2].set_xlabel('Day of Simulation')
    axes[1, 0].plot(days, results['employer_balance'], color='green')
    axes[1, 0].set_title('Employer Balance Over Time')
    axes[1, 0].set_xlabel('Day of Simulation')
    axes[1, 1].plot(days, results['union_balance'], color='orange')
    axes[1, 1].set_title('Union Strike Fund Balance Over Time')
    axes[1, 1].set_xlabel('Day of Simulation')
    # Hide the last subplot
    axes[1, 2].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Add explanation for strike day markers
    if strike_pattern != 'indefinite' and strike_days:
        st.markdown("""
        **Note:** Orange dashed lines indicate scheduled strike days based on your selected pattern.
        """)

    # --- Network Statistics ---
    st.subheader("🌐 Network Statistics")
    st.markdown("""
    Overview of the network structures used in this simulation.
    """)
    
    # Display network statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Union Network Size", f"{sim.union_network.number_of_nodes()} nodes")
    with col2:
        st.metric("Union Network Edges", f"{sim.union_network.number_of_edges()} edges")
    with col3:
        st.metric("Employer Network Size", f"{sim.employer_network.number_of_nodes()} nodes")
    with col4:
        st.metric("Employer Network Edges", f"{sim.employer_network.number_of_edges()} edges")
    
    # Calculate additional network metrics
    union_density = nx.density(sim.union_network)
    employer_density = nx.density(sim.employer_network)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Union Network Density", f"{union_density:.4f}")
    with col2:
        st.metric("Employer Network Density", f"{employer_density:.4f}")
    
    # Show network connectivity information
    union_components = nx.number_connected_components(sim.union_network)
    employer_components = nx.number_connected_components(sim.employer_network)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Union Connected Components", union_components)
    with col2:
        st.metric("Employer Connected Components", employer_components)


    # --- Display Calendar & Strike Pattern Info ---
    st.subheader("📅 Calendar & Strike Pattern")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Start Date", start_date.strftime('%Y-%m-%d'))
    with col2:
        st.metric("Duration", f"{duration} days")
    with col3:
        st.metric("Working Days", f"{len(working_days)} days/week")
    with col4:
        pattern_display = {
            'indefinite': 'Every working day',
            'once_a_week': 'Every Monday',
            'once_per_month': 'Every 1st of the month',
            'weekly_escalation': f'Week 1: {weekly_escalation_start} day(s), escalating weekly'
        }
        st.metric("Strike Pattern", pattern_display[strike_pattern])
    
    # Show working days
    st.write(f"**Working days:** {', '.join(sorted(working_days))}")
    
    # Show strike pattern details
    if strike_pattern == 'weekly_escalation':
        st.write(f"**Weekly escalation:** Starts with {weekly_escalation_start} strike day(s) per week, increases by 1 each week (max 5 days)")
    elif strike_pattern == 'once_a_week':
        st.write("**Weekly strikes:** Every Monday (if it's a working day)")
    elif strike_pattern == 'once_per_month':
        st.write("**Monthly strikes:** Every 1st of the month (if it's a working day)")
    else:
        st.write("**Indefinite strikes:** Workers can strike on any working day based on their morale")

    # --- Network Visualizations ---
    st.subheader("🌐 Network Visualizations")
    st.markdown("""
    Visual representation of the employer and union networks at the beginning and end of the simulation.
    Node colors indicate worker states: 🔴 Striking, 🔵 Working, 🟠 Union Committee, 🟢 Manager
    """)
    
    # Check if we have worker states data
    if results['worker_states'] and len(results['worker_states']) > 0:
        # Create visualizations for first and final timesteps
        first_timestep = 0
        final_timestep = len(results['worker_states']) - 1
        
        # First timestep visualization
        st.markdown("### 📊 First Timestep (Day 0)")
        try:
            fig_first = sim.visualize_networks_at_timestep(first_timestep)
            if fig_first:
                st.pyplot(fig_first)
                plt.close(fig_first)
            else:
                st.warning("Could not generate first timestep visualization")
        except Exception as e:
            st.error(f"Error generating first timestep visualization: {e}")
        
        # Final timestep visualization
        st.markdown("### 📊 Final Timestep (Day {})".format(final_timestep))
        try:
            fig_final = sim.visualize_networks_at_timestep(final_timestep)
            if fig_final:
                st.pyplot(fig_final)
                plt.close(fig_final)
            else:
                st.warning("Could not generate final timestep visualization")
        except Exception as e:
            st.error(f"Error generating final timestep visualization: {e}")
        
        # Add explanation of the visualizations
        st.markdown("""
        **Network Visualization Guide:**
        - **Left panel (Union Network):** Shows union membership and committee structure
        - **Right panel (Employer Network):** Shows organizational hierarchy and reporting relationships
        - **Node sizes:** Managers are slightly larger than regular workers
        - **Node colors:** Red = Striking, Blue = Working, Orange = Union Committee
        - **Edges:** Represent social/organizational connections that influence morale spread
        """)
        
    else:
        st.warning("No worker state data available for network visualization")

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