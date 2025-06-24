#run the model
import settings
from strikesim import StrikeSimulation
import pandas as pd
import os
from datetime import datetime

def main():
    """Main function to run the StrikeSim simulation"""
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('output', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert settings to dictionary
    settings_dict = {key: value for key, value in vars(settings).items() 
                    if not key.startswith('__')}
    
    print("Initializing StrikeSim simulation...")
    print(f"Start date: {settings.start_date}")
    print(f"Duration: {settings.duration} days")
    print(f"Number of workers: {settings.num_workers}")
    
    # Create simulation instance
    simulation = StrikeSimulation(settings_dict)
    
    print("\nRunning simulation...")
    # Run the simulation
    results = simulation.run_simulation()
    
    print(f"Simulation completed after {len(results['dates'])} days")
    
    # Analyze results
    analysis = simulation.analyze_results()
    print(f"\nSimulation Outcome: {analysis['outcome']}")
    print(f"Final striking workers: {analysis['final_striking_workers']}")
    print(f"Final working workers: {analysis['final_working_workers']}")
    print(f"Average morale: {analysis['average_morale']:.3f}")
    print(f"Total concessions: ${analysis['total_concessions']:.2f}")
    print(f"Total strike pay distributed: ${analysis['total_strike_pay']:.2f}")
    
    # Save the data
    print("\nSaving data...")
    h5_path = os.path.join(output_dir, 'strikesim_data.h5')
    csv_path = os.path.join(output_dir, 'strikesim_summary.csv')
    simulation.save_to_hdf5(h5_path)
    summary_df = simulation.save_summary_to_csv(csv_path)
    
    # Visualize the data
    print("Generating visualizations...")
    networks_path = os.path.join(output_dir, 'networks.png')
    timeseries_path = os.path.join(output_dir, 'time_series.png')
    simulation.visualize_networks(networks_path)
    simulation.visualize_time_series(timeseries_path)
    
    print("\nSimulation complete! Files generated:")
    print(f"- {h5_path} (full time series data)")
    print(f"- {csv_path} (summary statistics)")
    print(f"- {networks_path} (network visualizations)")
    print(f"- {timeseries_path} (time series plots)")
    
    return simulation, results, analysis

def run_monte_carlo():
    """Run Monte Carlo simulation"""
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('output', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    print("Running Monte Carlo simulation...")
    
    # Convert settings to dictionary
    settings_dict = {key: value for key, value in vars(settings).items() 
                    if not key.startswith('__')}
    
    # Create simulation instance
    simulation = StrikeSimulation(settings_dict)
    
    # Run Monte Carlo
    mc_results = simulation.run_monte_carlo(settings.monte_carlo_simulations)
    
    # Save Monte Carlo results
    mc_csv_path = os.path.join(output_dir, 'monte_carlo_results.csv')
    mc_results.to_csv(mc_csv_path, index=False)
    
    # Analyze Monte Carlo results
    print(f"\nMonte Carlo Results ({settings.monte_carlo_simulations} simulations):")
    print(f"Strike collapsed: {(mc_results['outcome'] == 'strike_collapsed').sum()}")
    print(f"Employer conceded: {(mc_results['outcome'] == 'employer_conceded').sum()}")
    print(f"Ongoing: {(mc_results['outcome'] == 'ongoing').sum()}")
    
    print(f"Average final striking workers: {mc_results['final_striking_workers'].mean():.1f}")
    print(f"Average final employer balance: ${mc_results['final_employer_balance'].mean():.2f}")
    print(f"Average final union balance: ${mc_results['final_union_balance'].mean():.2f}")
    
    print(f"Monte Carlo results saved to {mc_csv_path}")
    return mc_results

if __name__ == "__main__":
    # Run single simulation
    simulation, results, analysis = main()
    
    # Uncomment to run Monte Carlo simulation
    # mc_results = run_monte_carlo()