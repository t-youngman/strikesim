#!/usr/bin/env python3
"""
Parameter sweep script for StrikeSim model.
Runs the model with different parameter combinations and outputs key metrics.
"""

import numpy as np
import pandas as pd
from strikesim import StrikeSimulation
import os
from datetime import datetime
import itertools
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def load_emory_settings():
    """Load Emory settings and convert to dictionary"""
    import Emory_settings
    
    # Convert settings to dictionary
    settings_dict = {key: value for key, value in vars(Emory_settings).items() 
                    if not key.startswith('_')}
    
    return settings_dict

def run_parameter_sweep():
    """Run parameter sweep with specified ranges"""
    
    # Load base settings
    base_settings = load_emory_settings()
    
    # Define parameter ranges
    inflation_range = np.arange(0.0, 0.21, 0.01)  # 0 to 0.2 in 0.01 steps
    belt_tightening_range = np.arange(-1.0, 1.01, 0.01)  # -1 to 1 in 0.01 steps
    sigmoid_gamma_range = np.arange(0.0, 1.01, 0.01)  # 0 to 1 in 0.01 steps
    private_morale_alpha_range = np.arange(0.0, 1.01, 0.01)  # 0 to 1 in 0.01 steps
    
    # Calculate total number of combinations
    total_combinations = (len(inflation_range) * len(belt_tightening_range) * 
                         len(sigmoid_gamma_range) * len(private_morale_alpha_range))
    
    print(f"Parameter sweep configuration:")
    print(f"  Inflation: {len(inflation_range)} values from {inflation_range[0]} to {inflation_range[-1]}")
    print(f"  Belt tightening: {len(belt_tightening_range)} values from {belt_tightening_range[0]} to {belt_tightening_range[-1]}")
    print(f"  Sigmoid gamma: {len(sigmoid_gamma_range)} values from {sigmoid_gamma_range[0]} to {sigmoid_gamma_range[-1]}")
    print(f"  Private morale alpha: {len(private_morale_alpha_range)} values from {private_morale_alpha_range[0]} to {private_morale_alpha_range[-1]}")
    print(f"  Total combinations: {total_combinations:,}")
    print()
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'parameter_sweep_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save base settings
    with open(os.path.join(output_dir, 'base_settings.py'), 'w') as f:
        f.write("# Base settings used for parameter sweep\n")
        for key, value in base_settings.items():
            f.write(f"{key} = {value}\n")
    
    # Initialize results storage
    results = []
    
    # Create progress bar
    pbar = tqdm(total=total_combinations, desc="Running parameter sweep")
    
    # Run parameter combinations
    for inflation in inflation_range:
        for belt_tightening in belt_tightening_range:
            for sigmoid_gamma in sigmoid_gamma_range:
                for private_morale_alpha in private_morale_alpha_range:
                    
                    # Update settings with current parameter values
                    current_settings = base_settings.copy()
                    current_settings['inflation'] = float(inflation)
                    current_settings['belt_tightening'] = float(belt_tightening)
                    current_settings['sigmoid_gamma'] = float(sigmoid_gamma)
                    current_settings['private_morale_alpha'] = float(private_morale_alpha)
                    
                    try:
                        # Create and run simulation
                        sim = StrikeSimulation(current_settings)
                        sim_data = sim.run_simulation()
                        
                        # Extract final metrics
                        final_striking_workers = sim_data['striking_workers'][-1] if sim_data['striking_workers'] else 0
                        final_average_morale = sim_data['average_morale'][-1] if sim_data['average_morale'] else 0.0
                        
                        # Store results
                        result = {
                            'inflation': inflation,
                            'belt_tightening': belt_tightening,
                            'sigmoid_gamma': sigmoid_gamma,
                            'private_morale_alpha': private_morale_alpha,
                            'final_striking_workers': final_striking_workers,
                            'final_average_morale': final_average_morale,
                            'total_days': len(sim_data['dates']),
                            'max_striking_workers': max(sim_data['striking_workers']) if sim_data['striking_workers'] else 0,
                            'min_striking_workers': min(sim_data['striking_workers']) if sim_data['striking_workers'] else 0,
                            'final_employer_balance': sim_data['employer_balance'][-1] if sim_data['employer_balance'] else 0,
                            'final_union_balance': sim_data['union_balance'][-1] if sim_data['union_balance'] else 0
                        }
                        
                        results.append(result)
                        
                    except Exception as e:
                        print(f"\nError in simulation with parameters:")
                        print(f"  inflation={inflation}, belt_tightening={belt_tightening}")
                        print(f"  sigmoid_gamma={sigmoid_gamma}, private_morale_alpha={private_morale_alpha}")
                        print(f"  Error: {e}")
                        
                        # Store error result
                        result = {
                            'inflation': inflation,
                            'belt_tightening': belt_tightening,
                            'sigmoid_gamma': sigmoid_gamma,
                            'private_morale_alpha': private_morale_alpha,
                            'final_striking_workers': -1,  # Error indicator
                            'final_average_morale': -1,    # Error indicator
                            'total_days': 0,
                            'max_striking_workers': -1,
                            'min_striking_workers': -1,
                            'final_employer_balance': 0,
                            'final_union_balance': 0,
                            'error': str(e)
                        }
                        results.append(result)
                    
                    # Update progress bar
                    pbar.update(1)
    
    pbar.close()
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_file = os.path.join(output_dir, 'parameter_sweep_results.csv')
    results_df.to_csv(results_file, index=False)
    
    # Create summary statistics
    summary_file = os.path.join(output_dir, 'parameter_sweep_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Parameter Sweep Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total combinations: {total_combinations:,}\n")
        f.write(f"Successful runs: {len(results_df[results_df['final_striking_workers'] >= 0])}\n")
        f.write(f"Failed runs: {len(results_df[results_df['final_striking_workers'] < 0])}\n\n")
        
        # Filter out error runs for statistics
        valid_results = results_df[results_df['final_striking_workers'] >= 0]
        
        if len(valid_results) > 0:
            f.write("Results Statistics (excluding errors):\n")
            f.write("-" * 40 + "\n")
            f.write(f"Final striking workers:\n")
            f.write(f"  Mean: {valid_results['final_striking_workers'].mean():.2f}\n")
            f.write(f"  Std: {valid_results['final_striking_workers'].std():.2f}\n")
            f.write(f"  Min: {valid_results['final_striking_workers'].min()}\n")
            f.write(f"  Max: {valid_results['final_striking_workers'].max()}\n\n")
            
            f.write(f"Final average morale:\n")
            f.write(f"  Mean: {valid_results['final_average_morale'].mean():.4f}\n")
            f.write(f"  Std: {valid_results['final_average_morale'].std():.4f}\n")
            f.write(f"  Min: {valid_results['final_average_morale'].min():.4f}\n")
            f.write(f"  Max: {valid_results['final_average_morale'].max():.4f}\n\n")
            
            # Find parameter combinations with highest/lowest striking workers
            max_striking_idx = valid_results['final_striking_workers'].idxmax()
            min_striking_idx = valid_results['final_striking_workers'].idxmin()
            
            f.write("Parameter combinations with maximum striking workers:\n")
            max_row = valid_results.loc[max_striking_idx]
            f.write(f"  Inflation: {max_row['inflation']}\n")
            f.write(f"  Belt tightening: {max_row['belt_tightening']}\n")
            f.write(f"  Sigmoid gamma: {max_row['sigmoid_gamma']}\n")
            f.write(f"  Private morale alpha: {max_row['private_morale_alpha']}\n")
            f.write(f"  Final striking workers: {max_row['final_striking_workers']}\n")
            f.write(f"  Final average morale: {max_row['final_average_morale']:.4f}\n\n")
            
            f.write("Parameter combinations with minimum striking workers:\n")
            min_row = valid_results.loc[min_striking_idx]
            f.write(f"  Inflation: {min_row['inflation']}\n")
            f.write(f"  Belt tightening: {min_row['belt_tightening']}\n")
            f.write(f"  Sigmoid gamma: {min_row['sigmoid_gamma']}\n")
            f.write(f"  Private morale alpha: {min_row['private_morale_alpha']}\n")
            f.write(f"  Final striking workers: {min_row['final_striking_workers']}\n")
            f.write(f"  Final average morale: {min_row['final_average_morale']:.4f}\n\n")
    
    # Create correlation analysis
    correlation_file = os.path.join(output_dir, 'parameter_correlations.csv')
    if len(valid_results) > 0:
        # Calculate correlations with key metrics
        param_cols = ['inflation', 'belt_tightening', 'sigmoid_gamma', 'private_morale_alpha']
        metric_cols = ['final_striking_workers', 'final_average_morale']
        
        correlations = valid_results[param_cols + metric_cols].corr()
        correlations.to_csv(correlation_file)
        
        # Add correlation summary to summary file
        with open(summary_file, 'a') as f:
            f.write("Parameter Correlations:\n")
            f.write("-" * 25 + "\n")
            for param in param_cols:
                f.write(f"{param} vs final_striking_workers: {correlations.loc[param, 'final_striking_workers']:.4f}\n")
                f.write(f"{param} vs final_average_morale: {correlations.loc[param, 'final_average_morale']:.4f}\n")
            f.write("\n")
    
    print(f"\nParameter sweep completed!")
    print(f"Results saved to: {output_dir}")
    print(f"  - Full results: {results_file}")
    print(f"  - Summary: {summary_file}")
    if len(valid_results) > 0:
        print(f"  - Correlations: {correlation_file}")
    
    return results_df, output_dir

def create_visualization_script(results_df, output_dir):
    """Create a script to visualize the parameter sweep results"""
    
    viz_script = os.path.join(output_dir, 'visualize_results.py')
    
    script_content = '''#!/usr/bin/env python3
"""
Visualization script for parameter sweep results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load results
results_df = pd.read_csv('parameter_sweep_results.csv')

# Filter out error runs
valid_results = results_df[results_df['final_striking_workers'] >= 0]

if len(valid_results) == 0:
    print("No valid results to visualize")
    exit()

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Inflation vs Final Striking Workers
axes[0, 0].scatter(valid_results['inflation'], valid_results['final_striking_workers'], alpha=0.6)
axes[0, 0].set_xlabel('Inflation')
axes[0, 0].set_ylabel('Final Striking Workers')
axes[0, 0].set_title('Inflation vs Final Striking Workers')
axes[0, 0].grid(True, alpha=0.3)

# 2. Belt Tightening vs Final Striking Workers
axes[0, 1].scatter(valid_results['belt_tightening'], valid_results['final_striking_workers'], alpha=0.6)
axes[0, 1].set_xlabel('Belt Tightening')
axes[0, 1].set_ylabel('Final Striking Workers')
axes[0, 1].set_title('Belt Tightening vs Final Striking Workers')
axes[0, 1].grid(True, alpha=0.3)

# 3. Sigmoid Gamma vs Final Striking Workers
axes[1, 0].scatter(valid_results['sigmoid_gamma'], valid_results['final_striking_workers'], alpha=0.6)
axes[1, 0].set_xlabel('Sigmoid Gamma')
axes[1, 0].set_ylabel('Final Striking Workers')
axes[1, 0].set_title('Sigmoid Gamma vs Final Striking Workers')
axes[1, 0].grid(True, alpha=0.3)

# 4. Private Morale Alpha vs Final Striking Workers
axes[1, 1].scatter(valid_results['private_morale_alpha'], valid_results['final_striking_workers'], alpha=0.6)
axes[1, 1].set_xlabel('Private Morale Alpha')
axes[1, 1].set_ylabel('Final Striking Workers')
axes[1, 1].set_title('Private Morale Alpha vs Final Striking Workers')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('parameter_sweep_scatter_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# Create heatmap of correlations
plt.figure(figsize=(10, 8))
param_cols = ['inflation', 'belt_tightening', 'sigmoid_gamma', 'private_morale_alpha']
metric_cols = ['final_striking_workers', 'final_average_morale']
corr_data = valid_results[param_cols + metric_cols].corr()

sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.3f')
plt.title('Parameter Correlation Matrix')
plt.tight_layout()
plt.savefig('parameter_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualizations saved as:")
print("- parameter_sweep_scatter_plots.png")
print("- parameter_correlation_heatmap.png")
'''
    
    with open(viz_script, 'w') as f:
        f.write(script_content)
    
    print(f"  - Visualization script: {viz_script}")

if __name__ == "__main__":
    # Run the parameter sweep
    results_df, output_dir = run_parameter_sweep()
    
    # Create visualization script
    create_visualization_script(results_df, output_dir)
    
    print(f"\nTo visualize results, run:")
    print(f"cd {output_dir}")
    print(f"python visualize_results.py") 