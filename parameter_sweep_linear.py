#!/usr/bin/env python3
"""
Parameter sweep script for StrikeSim model - Linear Morale Specification.
Runs the model with different linear morale parameter combinations and outputs key metrics.
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

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def load_emory_settings():
    """Load Emory settings and convert to dictionary"""
    import settings_Emory as settings_Emory
    
    # Convert settings to dictionary
    settings_dict = {key: value for key, value in vars(settings_Emory).items() 
                    if not key.startswith('_')}
    
    return settings_dict

def create_heatmap_visualizations(results_df, output_dir):
    """Create heatmap visualizations for linear morale parameter interactions"""
    
    # Filter out error runs
    valid_results = results_df[results_df['final_striking_workers'] >= 0].copy()
    
    if len(valid_results) == 0:
        print("No valid results to visualize")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Linear Morale Parameter Interaction Heatmaps', fontsize=16, fontweight='bold')
    
    # Define parameter pairs for heatmaps
    param_pairs = [
        ('linear_alpha', 'linear_beta'),
        ('linear_alpha', 'linear_gamma'),
        ('linear_alpha', 'linear_phi'),
        ('linear_beta', 'linear_gamma'),
        ('linear_beta', 'linear_phi'),
        ('linear_gamma', 'linear_phi')
    ]
    
    # Create heatmaps for each parameter pair
    for idx, (param1, param2) in enumerate(param_pairs):
        ax = axes[idx // 3, idx % 3]
        
        # Create pivot table for heatmap
        pivot_data = valid_results.pivot_table(
            values='final_striking_workers',
            index=param1,
            columns=param2,
            aggfunc='mean'
        )
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=False, cmap='RdYlBu_r', 
                   cbar_kws={'label': 'Final Striking Workers'}, ax=ax)
        ax.set_title(f'{param1.replace("_", " ").title()} vs {param2.replace("_", " ").title()}')
        ax.set_xlabel(param2.replace('_', ' ').title())
        ax.set_ylabel(param1.replace('_', ' ').title())
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'linear_morale_parameter_interaction_heatmaps.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create morale heatmaps
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Linear Morale Parameter Interaction Heatmaps - Average Morale', fontsize=16, fontweight='bold')
    
    for idx, (param1, param2) in enumerate(param_pairs):
        ax = axes[idx // 3, idx % 3]
        
        # Create pivot table for heatmap
        pivot_data = valid_results.pivot_table(
            values='final_average_morale',
            index=param1,
            columns=param2,
            aggfunc='mean'
        )
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=False, cmap='RdYlGn', 
                   cbar_kws={'label': 'Final Average Morale'}, ax=ax)
        ax.set_title(f'{param1.replace("_", " ").title()} vs {param2.replace("_", " ").title()}')
        ax.set_xlabel(param2.replace('_', ' ').title())
        ax.set_ylabel(param1.replace('_', ' ').title())
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'linear_morale_parameter_interaction_heatmaps_morale.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_parameter_effect_plots(results_df, output_dir):
    """Create plots showing the effect of individual linear morale parameters"""
    
    # Filter out error runs
    valid_results = results_df[results_df['final_striking_workers'] >= 0].copy()
    
    if len(valid_results) == 0:
        print("No valid results to visualize")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Linear Morale Parameter Effects', fontsize=16, fontweight='bold')
    
    parameters = ['linear_alpha', 'linear_beta', 'linear_gamma', 'linear_phi']
    titles = ['Linear Alpha (Wage Gap Weight)', 'Linear Beta (Depletion Weight)', 
              'Linear Gamma (Current Morale Weight)', 'Linear Phi (Sigmoid Parameter)']
    
    for idx, (param, title) in enumerate(zip(parameters, titles)):
        ax = axes[idx // 2, idx % 2]
        
        # Create box plot
        sns.boxplot(data=valid_results, x=param, y='final_striking_workers', ax=ax)
        ax.set_title(f'{title} vs Final Striking Workers')
        ax.set_xlabel(title)
        ax.set_ylabel('Final Striking Workers')
        
        # Rotate x-axis labels if needed
        if len(valid_results[param].unique()) > 10:
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'linear_morale_parameter_effects_boxplots.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create scatter plots with regression lines
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Linear Morale Parameter Effects with Regression Lines', fontsize=16, fontweight='bold')
    
    for idx, (param, title) in enumerate(zip(parameters, titles)):
        ax = axes[idx // 2, idx % 2]
        
        # Create scatter plot with regression line
        sns.regplot(data=valid_results, x=param, y='final_striking_workers', 
                   scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'}, ax=ax)
        ax.set_title(f'{title} vs Final Striking Workers')
        ax.set_xlabel(title)
        ax.set_ylabel('Final Striking Workers')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'linear_morale_parameter_effects_regression.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_3d_visualizations(results_df, output_dir):
    """Create 3D scatter plots for linear morale parameter interactions"""
    
    # Filter out error runs
    valid_results = results_df[results_df['final_striking_workers'] >= 0].copy()
    
    if len(valid_results) == 0:
        print("No valid results to visualize")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    
    # Create 3D scatter plots
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Alpha vs Beta vs Striking Workers
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    scatter1 = ax1.scatter(valid_results['linear_alpha'], 
                          valid_results['linear_beta'], 
                          valid_results['final_striking_workers'],
                          c=valid_results['final_striking_workers'], 
                          cmap='viridis', alpha=0.6)
    ax1.set_xlabel('Linear Alpha')
    ax1.set_ylabel('Linear Beta')
    ax1.set_zlabel('Final Striking Workers')
    ax1.set_title('Alpha vs Beta vs Striking Workers')
    plt.colorbar(scatter1, ax=ax1, shrink=0.5)
    
    # Plot 2: Alpha vs Gamma vs Striking Workers
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    scatter2 = ax2.scatter(valid_results['linear_alpha'], 
                          valid_results['linear_gamma'], 
                          valid_results['final_striking_workers'],
                          c=valid_results['final_striking_workers'], 
                          cmap='plasma', alpha=0.6)
    ax2.set_xlabel('Linear Alpha')
    ax2.set_ylabel('Linear Gamma')
    ax2.set_zlabel('Final Striking Workers')
    ax2.set_title('Alpha vs Gamma vs Striking Workers')
    plt.colorbar(scatter2, ax=ax2, shrink=0.5)
    
    # Plot 3: Beta vs Gamma vs Striking Workers
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    scatter3 = ax3.scatter(valid_results['linear_beta'], 
                          valid_results['linear_gamma'], 
                          valid_results['final_striking_workers'],
                          c=valid_results['final_striking_workers'], 
                          cmap='inferno', alpha=0.6)
    ax3.set_xlabel('Linear Beta')
    ax3.set_ylabel('Linear Gamma')
    ax3.set_zlabel('Final Striking Workers')
    ax3.set_title('Beta vs Gamma vs Striking Workers')
    plt.colorbar(scatter3, ax=ax3, shrink=0.5)
    
    # Plot 4: Alpha vs Beta vs Morale
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    scatter4 = ax4.scatter(valid_results['linear_alpha'], 
                          valid_results['linear_beta'], 
                          valid_results['final_average_morale'],
                          c=valid_results['final_average_morale'], 
                          cmap='RdYlGn', alpha=0.6)
    ax4.set_xlabel('Linear Alpha')
    ax4.set_ylabel('Linear Beta')
    ax4.set_zlabel('Final Average Morale')
    ax4.set_title('Alpha vs Beta vs Morale')
    plt.colorbar(scatter4, ax=ax4, shrink=0.5)
    
    # Plot 5: Gamma vs Phi vs Striking Workers
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    scatter5 = ax5.scatter(valid_results['linear_gamma'], 
                          valid_results['linear_phi'], 
                          valid_results['final_striking_workers'],
                          c=valid_results['final_striking_workers'], 
                          cmap='coolwarm', alpha=0.6)
    ax5.set_xlabel('Linear Gamma')
    ax5.set_ylabel('Linear Phi')
    ax5.set_zlabel('Final Striking Workers')
    ax5.set_title('Gamma vs Phi vs Striking Workers')
    plt.colorbar(scatter5, ax=ax5, shrink=0.5)
    
    # Plot 6: All parameters vs Striking Workers (using color for 4th parameter)
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    scatter6 = ax6.scatter(valid_results['linear_alpha'], 
                          valid_results['linear_beta'], 
                          valid_results['final_striking_workers'],
                          c=valid_results['linear_gamma'], 
                          cmap='viridis', alpha=0.6)
    ax6.set_xlabel('Linear Alpha')
    ax6.set_ylabel('Linear Beta')
    ax6.set_zlabel('Final Striking Workers')
    ax6.set_title('Alpha vs Beta vs Striking Workers\n(Color: Linear Gamma)')
    plt.colorbar(scatter6, ax=ax6, shrink=0.5, label='Linear Gamma')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'linear_morale_3d_parameter_visualizations.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_visualizations(results_df, output_dir):
    """Create summary visualizations of the linear morale parameter sweep results"""
    
    # Filter out error runs
    valid_results = results_df[results_df['final_striking_workers'] >= 0].copy()
    
    if len(valid_results) == 0:
        print("No valid results to visualize")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create summary figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Linear Morale Parameter Sweep Summary', fontsize=16, fontweight='bold')
    
    # Plot 1: Distribution of final striking workers
    ax1 = axes[0, 0]
    sns.histplot(valid_results['final_striking_workers'], bins=30, ax=ax1)
    ax1.set_title('Distribution of Final Striking Workers')
    ax1.set_xlabel('Final Striking Workers')
    ax1.set_ylabel('Frequency')
    
    # Plot 2: Distribution of final average morale
    ax2 = axes[0, 1]
    sns.histplot(valid_results['final_average_morale'], bins=30, ax=ax2)
    ax2.set_title('Distribution of Final Average Morale')
    ax2.set_xlabel('Final Average Morale')
    ax2.set_ylabel('Frequency')
    
    # Plot 3: Scatter plot of striking workers vs morale
    ax3 = axes[1, 0]
    scatter = ax3.scatter(valid_results['final_striking_workers'], 
                         valid_results['final_average_morale'],
                         c=valid_results['linear_alpha'], cmap='viridis', alpha=0.6)
    ax3.set_title('Striking Workers vs Average Morale\n(Color: Linear Alpha)')
    ax3.set_xlabel('Final Striking Workers')
    ax3.set_ylabel('Final Average Morale')
    plt.colorbar(scatter, ax=ax3, label='Linear Alpha')
    
    # Plot 4: Correlation heatmap of all parameters
    ax4 = axes[1, 1]
    param_cols = ['linear_alpha', 'linear_beta', 'linear_gamma', 'linear_phi']
    metric_cols = ['final_striking_workers', 'final_average_morale']
    correlation_data = valid_results[param_cols + metric_cols].corr()
    
    sns.heatmap(correlation_data, annot=True, cmap='RdBu_r', center=0, 
                square=True, ax=ax4, fmt='.3f')
    ax4.set_title('Linear Morale Parameter Correlations')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'linear_morale_parameter_sweep_summary.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def run_parameter_sweep():
    """Run parameter sweep with linear morale specification parameters"""
    
    # Load base settings
    base_settings = load_emory_settings()
    
    # Set morale specification to linear
    base_settings['morale_specification'] = 'linear'
    
    # Define linear morale parameter ranges
    linear_alpha_range = np.arange(0.1, 1.01, 0.1)  # 0.1 to 1.01 in 0.1 steps
    linear_beta_range = np.arange(0.1, 1.01, 0.1)   # 0.1 to 1.01 in 0.1 steps
    linear_gamma_range = np.arange(0.1, 1.01, 0.1)  # 0.2 to 1.01 in 0.1 steps
    linear_phi_range = np.arange(0.1, 1.01, 0.1)    # 0.1 to 1.01 in 0.1 steps
    
    # Calculate total number of combinations
    total_combinations = (len(linear_alpha_range) * len(linear_beta_range) * 
                         len(linear_gamma_range) * len(linear_phi_range))
    
    print(f"Linear Morale Parameter sweep configuration:")
    print(f"  Linear Alpha: {len(linear_alpha_range)} values from {linear_alpha_range[0]} to {linear_alpha_range[-1]}")
    print(f"  Linear Beta: {len(linear_beta_range)} values from {linear_beta_range[0]} to {linear_beta_range[-1]}")
    print(f"  Linear Gamma: {len(linear_gamma_range)} values from {linear_gamma_range[0]} to {linear_gamma_range[-1]}")
    print(f"  Linear Phi: {len(linear_phi_range)} values from {linear_phi_range[0]} to {linear_phi_range[-1]}")
    print(f"  Total combinations: {total_combinations:,}")
    print(f"  Morale specification: {base_settings['morale_specification']}")
    print()
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'linear_morale_parameter_sweep_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save base settings
    with open(os.path.join(output_dir, 'base_settings.py'), 'w') as f:
        f.write("# Base settings used for linear morale parameter sweep\n")
        for key, value in base_settings.items():
            f.write(f"{key} = {value}\n")
    
    # Initialize results storage
    results = []
    
    # Create progress bar
    pbar = tqdm(total=total_combinations, desc="Running linear morale parameter sweep")
    
    # Run parameter combinations
    for linear_alpha in linear_alpha_range:
        for linear_beta in linear_beta_range:
            for linear_gamma in linear_gamma_range:
                for linear_phi in linear_phi_range:
                    
                    # Update settings with current parameter values
                    current_settings = base_settings.copy()
                    current_settings['linear_alpha'] = float(linear_alpha)
                    current_settings['linear_beta'] = float(linear_beta)
                    current_settings['linear_gamma'] = float(linear_gamma)
                    current_settings['linear_phi'] = float(linear_phi)
                    
                    try:
                        # Create and run simulation
                        sim = StrikeSimulation(current_settings)
                        sim_data = sim.run_simulation()
                        
                        # Extract final metrics
                        final_striking_workers = sim_data['striking_workers'][-1] if sim_data['striking_workers'] else 0
                        final_average_morale = sim_data['average_morale'][-1] if sim_data['average_morale'] else 0.0
                        
                        # Store results
                        result = {
                            'linear_alpha': linear_alpha,
                            'linear_beta': linear_beta,
                            'linear_gamma': linear_gamma,
                            'linear_phi': linear_phi,
                            'final_striking_workers': final_striking_workers,
                            'final_average_morale': final_average_morale,
                            'max_striking_workers': max(sim_data['striking_workers']) if sim_data['striking_workers'] else 0,
                            'min_striking_workers': min(sim_data['striking_workers']) if sim_data['striking_workers'] else 0,
                        }
                        
                        results.append(result)
                        
                    except Exception as e:
                        print(f"\nError in simulation with parameters:")
                        print(f"  linear_alpha={linear_alpha}, linear_beta={linear_beta}")
                        print(f"  linear_gamma={linear_gamma}, linear_phi={linear_phi}")
                        print(f"  Error: {e}")
                        
                        # Store error result
                        result = {
                            'linear_alpha': linear_alpha,
                            'linear_beta': linear_beta,
                            'linear_gamma': linear_gamma,
                            'linear_phi': linear_phi,
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
    results_file = os.path.join(output_dir, 'linear_morale_parameter_sweep_results.csv')
    results_df.to_csv(results_file, index=False)
    
    # Create summary statistics
    summary_file = os.path.join(output_dir, 'linear_morale_parameter_sweep_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Linear Morale Parameter Sweep Summary\n")
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
            f.write(f"  Linear Alpha: {max_row['linear_alpha']}\n")
            f.write(f"  Linear Beta: {max_row['linear_beta']}\n")
            f.write(f"  Linear Gamma: {max_row['linear_gamma']}\n")
            f.write(f"  Linear Phi: {max_row['linear_phi']}\n")
            f.write(f"  Final striking workers: {max_row['final_striking_workers']}\n")
            f.write(f"  Final average morale: {max_row['final_average_morale']:.4f}\n\n")
            
            f.write("Parameter combinations with minimum striking workers:\n")
            min_row = valid_results.loc[min_striking_idx]
            f.write(f"  Linear Alpha: {min_row['linear_alpha']}\n")
            f.write(f"  Linear Beta: {min_row['linear_beta']}\n")
            f.write(f"  Linear Gamma: {min_row['linear_gamma']}\n")
            f.write(f"  Linear Phi: {min_row['linear_phi']}\n")
            f.write(f"  Final striking workers: {min_row['final_striking_workers']}\n")
            f.write(f"  Final average morale: {min_row['final_average_morale']:.4f}\n\n")
    
    # Create correlation analysis
    correlation_file = os.path.join(output_dir, 'linear_morale_parameter_correlations.csv')
    if len(valid_results) > 0:
        # Calculate correlations with key metrics
        param_cols = ['linear_alpha', 'linear_beta', 'linear_gamma', 'linear_phi']
        metric_cols = ['final_striking_workers', 'final_average_morale']
        
        correlations = valid_results[param_cols + metric_cols].corr()
        correlations.to_csv(correlation_file)
        
        # Add correlation summary to summary file
        with open(summary_file, 'a') as f:
            f.write("Linear Morale Parameter Correlations:\n")
            f.write("-" * 35 + "\n")
            for param in param_cols:
                f.write(f"{param} vs final_striking_workers: {correlations.loc[param, 'final_striking_workers']:.4f}\n")
                f.write(f"{param} vs final_average_morale: {correlations.loc[param, 'final_average_morale']:.4f}\n")
            f.write("\n")
    
    print(f"\nLinear morale parameter sweep completed!")
    print(f"Results saved to: {output_dir}")
    print(f"  - Full results: {results_file}")
    print(f"  - Summary: {summary_file}")
    if len(valid_results) > 0:
        print(f"  - Correlations: {correlation_file}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    try:
        create_heatmap_visualizations(results_df, output_dir)
        print("  - Heatmap visualizations created")
        
        create_parameter_effect_plots(results_df, output_dir)
        print("  - Parameter effect plots created")
        
        create_3d_visualizations(results_df, output_dir)
        print("  - 3D visualizations created")
        
        create_summary_visualizations(results_df, output_dir)
        print("  - Summary visualizations created")
        
        print(f"\nAll visualizations saved to: {output_dir}")
        
    except Exception as e:
        print(f"Warning: Error creating visualizations: {e}")
    
    return results_df, output_dir

if __name__ == "__main__":
    # Run the linear morale parameter sweep
    results, output_directory = run_parameter_sweep()
    print(f"\nLinear morale parameter sweep completed successfully!")
    print(f"Check the output directory '{output_directory}' for all results and visualizations.") 