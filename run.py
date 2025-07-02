#!/usr/bin/env python3
"""
Example script to run the strike simulation with different network configurations.
"""

import settings.settings as settings
from strikesim import StrikeSimulation
import os
import shutil
from datetime import datetime

def create_output_directory(base_dir='output'):
    """Create output directory with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_dir, f'employer_{timestamp}')
    
    # Create the directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

def save_settings_to_output(output_dir):
    """Save a copy of the current settings to the output directory"""
    settings_file = os.path.join(output_dir, 'settings.py')
    
    # Read the original settings file
    with open('settings.py', 'r') as f:
        settings_content = f.read()
    
    # Write to output directory
    with open(settings_file, 'w') as f:
        f.write(settings_content)
    
    print(f"Settings saved to: {settings_file}")

def run_simulation_with_output(sim, output_dir, run_name):
    """Run a simulation and save all outputs to the specified directory"""
    print(f"=== {run_name} ===")
    
    # Create subdirectory for this run
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Run simulation
    results = sim.run_simulation()
    print(f"Simulation completed with {len(results['dates'])} days")
    
    # Save results
    summary_file = os.path.join(run_dir, 'strikesim_summary.csv')
    data_file = os.path.join(run_dir, 'strikesim_data.h5')
    networks_file = os.path.join(run_dir, 'networks.png')
    time_series_file = os.path.join(run_dir, 'time_series.png')
    
    sim.save_summary_to_csv(summary_file)
    sim.save_to_hdf5(data_file)
    sim.visualize_networks(networks_file)
    sim.visualize_time_series(time_series_file)
    
    # Analyze results
    analysis = sim.analyze_results()
    
    # Save analysis to text file
    analysis_file = os.path.join(run_dir, 'analysis.txt')
    with open(analysis_file, 'w') as f:
        f.write(f"Simulation Analysis for {run_name}\n")
        f.write("=" * 50 + "\n\n")
        for key, value in analysis.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Results saved to: {run_dir}")
    print(f"  - Summary: {summary_file}")
    print(f"  - Data: {data_file}")
    print(f"  - Networks: {networks_file}")
    print(f"  - Time Series: {time_series_file}")
    print(f"  - Analysis: {analysis_file}")
    print()
    
    return analysis

def main():
    # Create output directory with timestamp
    output_dir = create_output_directory()
    print(f"Output directory created: {output_dir}")
    print()
    
    # Save settings to output directory
    save_settings_to_output(output_dir)
    print()
    
    # Convert settings to dictionary
    settings_dict = {key: value for key, value in vars(settings).items() 
                    if not key.startswith('_')}
    
    # Create simulation instance
    sim = StrikeSimulation(settings_dict)
    
    # Show available networks
    print("Available network files:")
    available_networks = sim.get_available_networks()
    print(f"Employer networks: {available_networks['employers']}")
    print(f"Union networks: {available_networks['unions']}")
    print()
    
    # Example 1: Run with generated networks (default)
    settings_dict['employer_network_file'] = None
    settings_dict['union_network_file'] = None
    sim1 = StrikeSimulation(settings_dict)
    analysis1 = run_simulation_with_output(sim1, output_dir, 'generated_networks')
    
    # Example 2: Run with loaded employer network (if available)
    if available_networks['employers']:
        settings_dict['employer_network_file'] = available_networks['employers'][0]
        settings_dict['union_network_file'] = None
        sim2 = StrikeSimulation(settings_dict)
        analysis2 = run_simulation_with_output(sim2, output_dir, f'loaded_employer_{available_networks["employers"][0]}')
    
    # Example 3: Run with both networks loaded (if available)
    if available_networks['employers'] and available_networks['unions']:
        settings_dict['employer_network_file'] = available_networks['employers'][0]
        settings_dict['union_network_file'] = available_networks['unions'][0]
        sim3 = StrikeSimulation(settings_dict)
        analysis3 = run_simulation_with_output(sim3, output_dir, f'both_networks_{available_networks["employers"][0]}_{available_networks["unions"][0]}')
    
    # Create a summary report of all runs
    summary_report_file = os.path.join(output_dir, 'summary_report.txt')
    with open(summary_report_file, 'w') as f:
        f.write("StrikeSim Multi-Run Summary Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output directory: {output_dir}\n\n")
        
        f.write("Available Networks:\n")
        f.write(f"  Employer networks: {available_networks['employers']}\n")
        f.write(f"  Union networks: {available_networks['unions']}\n\n")
        
        f.write("Simulation Runs:\n")
        f.write("-" * 20 + "\n")
        
        # Add analysis for each run
        f.write("1. Generated Networks:\n")
        for key, value in analysis1.items():
            f.write(f"   {key}: {value}\n")
        f.write("\n")
        
        if available_networks['employers']:
            f.write(f"2. Loaded Employer Network ({available_networks['employers'][0]}):\n")
            for key, value in analysis2.items():
                f.write(f"   {key}: {value}\n")
            f.write("\n")
        
        if available_networks['employers'] and available_networks['unions']:
            f.write(f"3. Both Networks Loaded:\n")
            for key, value in analysis3.items():
                f.write(f"   {key}: {value}\n")
            f.write("\n")
    
    print(f"Summary report saved to: {summary_report_file}")
    print(f"\nAll outputs organized in: {output_dir}")

if __name__ == "__main__":
    main()