import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
from strikesim import StrikeSimulation
import settings
import networkx as nx
from PIL import Image
import io

# --- Helper: List available .gexf networks ---
def list_gexf_files(folder):
    files = []
    if os.path.exists(folder):
        for f in os.listdir(folder):
            if f.endswith('.gexf'):
                files.append(os.path.splitext(f)[0])
    return files

# --- Initialize session state ---
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'simulation_object' not in st.session_state:
    st.session_state.simulation_object = None
if 'current_timestep' not in st.session_state:
    st.session_state.current_timestep = 0

def create_network_figure(union_network, employer_network, worker_states, timestep):
    """Create network visualization using matplotlib"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Union network
    pos_union = nx.spring_layout(union_network, k=1, iterations=50)
    
    # Color nodes based on worker states
    node_colors_union = []
    node_sizes_union = []
    node_labels_union = {}
    
    for node in union_network.nodes():
        if node < len(worker_states):  # Worker node
            if worker_states[node] == 'striking':
                node_colors_union.append('#ff4444')  # Bright red
            elif worker_states[node] == 'not_striking':
                node_colors_union.append('#4444ff')  # Bright blue
            else:
                node_colors_union.append('#888888')  # Gray
            node_sizes_union.append(400)
            node_labels_union[node] = f'W{node}'
        else:  # Committee node
            node_colors_union.append('#ffaa00')  # Orange
            node_sizes_union.append(600)
            node_labels_union[node] = f'C{abs(node)}'
    
    nx.draw(union_network, pos_union, ax=ax1,
            node_color=node_colors_union, node_size=node_sizes_union,
            labels=node_labels_union, font_size=10, font_weight='bold',
            edge_color='gray', alpha=0.7, width=2)
    ax1.set_title(f'Union Network - Day {timestep}', fontsize=14, fontweight='bold')
    
    # Add legend for union network
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff4444', 
                  markersize=15, label='Striking Worker'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4444ff', 
                  markersize=15, label='Working Worker'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffaa00', 
                  markersize=15, label='Union Committee')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Employer network
    pos_employer = nx.spring_layout(employer_network, k=1, iterations=50)
    
    # Color nodes based on worker states
    node_colors_employer = []
    node_sizes_employer = []
    node_labels_employer = {}
    
    for node in employer_network.nodes():
        if node < len(worker_states):  # Worker node
            if worker_states[node] == 'striking':
                node_colors_employer.append('#ff4444')  # Bright red
            elif worker_states[node] == 'not_striking':
                node_colors_employer.append('#4444ff')  # Bright blue
            else:
                node_colors_employer.append('#888888')  # Gray
            node_sizes_employer.append(400)
            node_labels_employer[node] = f'W{node}'
        else:  # Management node
            node_colors_employer.append('#44ff44')  # Bright green
            node_sizes_employer.append(600)
            node_labels_employer[node] = f'M{node}'
    
    nx.draw(employer_network, pos_employer, ax=ax2,
            node_color=node_colors_employer, node_size=node_sizes_employer,
            labels=node_labels_employer, font_size=10, font_weight='bold',
            edge_color='gray', alpha=0.7, width=2)
    ax2.set_title(f'Employer Network - Day {timestep}', fontsize=14, fontweight='bold')
    
    # Add legend for employer network
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff4444', 
                  markersize=15, label='Striking Worker'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4444ff', 
                  markersize=15, label='Working Worker'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#44ff44', 
                  markersize=15, label='Management')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    return fig

st.set_page_config(page_title="StrikeSim Dashboard", layout="wide")
st.title("⚡ StrikeSim Interactive Dashboard")
st.markdown("""
Interactively explore how union structure and policy affect strike outcomes. Adjust parameters, run the simulation, and view results instantly!
""")

# --- Sidebar: Parameter Controls ---
st.sidebar.header("Simulation Parameters")

# --- Network Settings (Accordion) ---
with st.sidebar.expander("🌐 Workplace Networks", expanded=False):
    # Network selection
    employer_network_options = ['random'] + list_gexf_files('networks/employers')
    union_network_options = ['random'] + list_gexf_files('networks/unions')

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
        department_density = st.slider("Union density (fraction of workers)", 0.0, 1.0, settings.department_density)
        team_density = st.slider("Union team density", 0.0, 1.0, settings.team_density)
    else:
        bargaining_committee_size = None
        department_density = None
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
st.sidebar.header("Simulation Settings")
morale_specification = st.sidebar.selectbox("Morale specification", ["sigmoid", "linear", "no_motivation"], index=["sigmoid", "linear", "no_motivation"].index(settings.morale_specification))
duration = st.sidebar.slider("Simulation duration (days)", 10, 365, settings.duration)

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
    start_date=settings.start_date,
    working_days=settings.working_days,
    revenue_markup=revenue_markup,
    concession_threshold=concession_threshold,
    concession_policy=settings.concession_policy,
    strike_pay_policy=settings.strike_pay_policy,
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
    settings_dict['department_density'] = department_density
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
    st.session_state.current_timestep = 0

# --- Display results if available ---
if st.session_state.simulation_results is not None:
    results = st.session_state.simulation_results
    sim = st.session_state.simulation_object
    analysis = sim.analyze_results()

    # --- Show summary stats ---
    st.subheader("Simulation Outcome")
    col1, col2, col3 = st.columns(3)
    col1.metric("Outcome", analysis['outcome'])
    col2.metric("Final Striking Workers", analysis['final_striking_workers'])
    col3.metric("Final Working Workers", analysis['final_working_workers'])
    st.write(f"**Average morale:** {analysis['average_morale']:.3f}")
    st.write(f"**Average savings:** ${analysis['average_savings']:.2f}")
    st.write(f"**Total concessions:** ${analysis['total_concessions']:.2f}")
    st.write(f"**Total strike pay distributed:** ${analysis['total_strike_pay']:.2f}")

    # --- Time Series Plots ---
    st.subheader("Time Series Plots")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Create day numbers for x-axis
    days = list(range(len(results['striking_workers'])))
    
    axes[0, 0].plot(days, results['striking_workers'], label='Striking', color='red')
    axes[0, 0].plot(days, results['working_workers'], label='Working', color='blue')
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

    # --- Interactive Network Visualization ---
    st.subheader("🌐 Interactive Network Visualization")
    st.markdown("""
    Explore how the strike spreads through the union and employer networks. 
    Use the slider to step through each day of the simulation - no re-running needed!
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
    
    # Create slider for timestep selection
    max_timestep = len(results['striking_workers']) - 1
    selected_timestep = st.slider(
        "Select Day", 
        min_value=0, 
        max_value=max_timestep, 
        value=st.session_state.current_timestep,
        help="Step through each day of the simulation to see how the strike spreads through the networks"
    )
    
    # Update session state
    st.session_state.current_timestep = selected_timestep
    
    # Display network statistics for selected timestep
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Day", selected_timestep)
    with col2:
        st.metric("Striking Workers", results['striking_workers'][selected_timestep])
    with col3:
        st.metric("Working Workers", results['working_workers'][selected_timestep])
    
    # Create network visualization
    fig = create_network_figure(
        sim.union_network, 
        sim.employer_network, 
        results['worker_states'][selected_timestep], 
        selected_timestep
    )
    
    # Display network visualization
    st.pyplot(fig)
    
    # Add explanation of the visualization
    st.markdown("""
    **Network Visualization Guide:**
    - **Red nodes**: Workers currently on strike
    - **Blue nodes**: Workers currently working
    - **Orange nodes**: Union committee members (Union network only)
    - **Green nodes**: Management positions (Employer network only)
    - **Gray edges**: Network connections between workers and organizational structures
    
    The visualization shows how strike participation spreads through both the union's organizational structure 
    and the employer's hierarchical network over time. Hover over nodes for more details!
    """)
    
    # --- Create Animation Section ---
    st.subheader("🎬 Create Network Animation")
    st.markdown("Generate an animated GIF showing how the networks evolve over the entire simulation.")
    
    col1, col2 = st.columns(2)
    with col1:
        fps = st.slider("Animation Speed (FPS)", 1, 5, 2, help="Frames per second for the animation")
    with col2:
        max_frames = st.number_input("Max Frames (optional)", min_value=10, max_value=100, value=50, 
                                   help="Limit frames for smaller file size. Leave empty for all frames.")
    
    if st.button("🎬 Generate Animation", type="primary"):
        with st.spinner("Creating animation... This may take a moment..."):
            # Create temporary file for the animation
            with tempfile.NamedTemporaryFile(delete=False, suffix='.gif') as tmp_gif:
                animation_path = sim.create_network_animation(
                    save_path=tmp_gif.name,
                    fps=fps,
                    max_frames=max_frames if max_frames > 0 else None
                )
                
                if animation_path and os.path.exists(animation_path):
                    # Read the GIF file
                    with open(animation_path, 'rb') as f:
                        gif_data = f.read()
                    
                    # Display the animation
                    st.success("Animation created successfully!")
                    st.image(gif_data, caption="Network Evolution Animation")
                    
                    # Download button
                    st.download_button(
                        "📥 Download Animation GIF",
                        data=gif_data,
                        file_name='network_animation.gif',
                        mime='image/gif'
                    )
                    
                    # Clean up temporary file
                    os.unlink(animation_path)
                else:
                    st.error("Failed to create animation. Please try again.")

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