#!/usr/bin/env python3
"""
Example script to run the strike simulation with different network configurations.
"""

import settings
from strikesim import StrikeSimulation

def main():
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
    print("=== Running with generated networks ===")
    settings_dict['employer_network_file'] = None
    settings_dict['union_network_file'] = None
    sim1 = StrikeSimulation(settings_dict)
    results1 = sim1.run_simulation()
    print(f"Simulation completed with {len(results1['dates'])} days")
    print()
    
    # Example 2: Run with loaded employer network (if available)
    if available_networks['employers']:
        print("=== Running with loaded employer network ===")
        settings_dict['employer_network_file'] = available_networks['employers'][0]
        settings_dict['union_network_file'] = None
        sim2 = StrikeSimulation(settings_dict)
        results2 = sim2.run_simulation()
        print(f"Simulation completed with {len(results2['dates'])} days")
        print()
    
    # Example 3: Run with both networks loaded (if available)
    if available_networks['employers'] and available_networks['unions']:
        print("=== Running with both networks loaded ===")
        settings_dict['employer_network_file'] = available_networks['employers'][0]
        settings_dict['union_network_file'] = available_networks['unions'][0]
        sim3 = StrikeSimulation(settings_dict)
        results3 = sim3.run_simulation()
        print(f"Simulation completed with {len(results3['dates'])} days")
        print()
    
    # Save results and create visualizations
    print("=== Saving results ===")
    sim1.save_summary_to_csv('strikesim_summary.csv')
    sim1.save_to_hdf5('strikesim_data.h5')
    sim1.visualize_networks('networks.png')
    sim1.visualize_time_series('time_series.png')
    
    # Analyze results
    analysis = sim1.analyze_results()
    print("Simulation Analysis:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()