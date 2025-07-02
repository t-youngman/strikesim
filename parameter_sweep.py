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
    """Create heatmap visualizations for parameter interactions"""
    
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
    fig.suptitle('Parameter Interaction Heatmaps', fontsize=16, fontweight='bold')
    
    # Define parameter pairs for heatmaps
    param_pairs = [
        ('inflation', 'belt_tightening'),
        ('inflation', 'sigmoid_gamma'),
        ('inflation', 'private_morale_alpha'),
        ('belt_tightening', 'sigmoid_gamma'),
        ('belt_tightening', 'private_morale_alpha'),
        ('sigmoid_gamma', 'private_morale_alpha')
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
    plt.savefig(os.path.join(output_dir, 'parameter_interaction_heatmaps.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create morale heatmaps
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Parameter Interaction Heatmaps - Average Morale', fontsize=16, fontweight='bold')
    
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
    plt.savefig(os.path.join(output_dir, 'parameter_interaction_heatmaps_morale.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_parameter_effect_plots(results_df, output_dir):
    """Create plots showing the effect of individual parameters"""
    
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
    fig.suptitle('Individual Parameter Effects', fontsize=16, fontweight='bold')
    
    parameters = ['inflation', 'belt_tightening', 'sigmoid_gamma', 'private_morale_alpha']
    titles = ['Inflation', 'Belt Tightening', 'Sigmoid Gamma', 'Private Morale Alpha']
    
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
    plt.savefig(os.path.join(output_dir, 'parameter_effects_boxplots.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create scatter plots with regression lines
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Parameter Effects with Regression Lines', fontsize=16, fontweight='bold')
    
    for idx, (param, title) in enumerate(zip(parameters, titles)):
        ax = axes[idx // 2, idx % 2]
        
        # Create scatter plot with regression line
        sns.regplot(data=valid_results, x=param, y='final_striking_workers', 
                   scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'}, ax=ax)
        ax.set_title(f'{title} vs Final Striking Workers')
        ax.set_xlabel(title)
        ax.set_ylabel('Final Striking Workers')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_effects_regression.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_3d_visualizations(results_df, output_dir):
    """Create 3D scatter plots for parameter interactions"""
    
    # Filter out error runs
    valid_results = results_df[results_df['final_striking_workers'] >= 0].copy()
    
    if len(valid_results) == 0:
        print("No valid results to visualize")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    
    # Create 3D scatter plots
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Inflation vs Belt Tightening vs Striking Workers
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    scatter1 = ax1.scatter(valid_results['inflation'], 
                          valid_results['belt_tightening'], 
                          valid_results['final_striking_workers'],
                          c=valid_results['final_striking_workers'], 
                          cmap='viridis', alpha=0.6)
    ax1.set_xlabel('Inflation')
    ax1.set_ylabel('Belt Tightening')
    ax1.set_zlabel('Final Striking Workers')
    ax1.set_title('Inflation vs Belt Tightening vs Striking Workers')
    plt.colorbar(scatter1, ax=ax1, shrink=0.5)
    
    # Plot 2: Inflation vs Sigmoid Gamma vs Striking Workers
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    scatter2 = ax2.scatter(valid_results['inflation'], 
                          valid_results['sigmoid_gamma'], 
                          valid_results['final_striking_workers'],
                          c=valid_results['final_striking_workers'], 
                          cmap='plasma', alpha=0.6)
    ax2.set_xlabel('Inflation')
    ax2.set_ylabel('Sigmoid Gamma')
    ax2.set_zlabel('Final Striking Workers')
    ax2.set_title('Inflation vs Sigmoid Gamma vs Striking Workers')
    plt.colorbar(scatter2, ax=ax2, shrink=0.5)
    
    # Plot 3: Belt Tightening vs Sigmoid Gamma vs Striking Workers
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    scatter3 = ax3.scatter(valid_results['belt_tightening'], 
                          valid_results['sigmoid_gamma'], 
                          valid_results['final_striking_workers'],
                          c=valid_results['final_striking_workers'], 
                          cmap='inferno', alpha=0.6)
    ax3.set_xlabel('Belt Tightening')
    ax3.set_ylabel('Sigmoid Gamma')
    ax3.set_zlabel('Final Striking Workers')
    ax3.set_title('Belt Tightening vs Sigmoid Gamma vs Striking Workers')
    plt.colorbar(scatter3, ax=ax3, shrink=0.5)
    
    # Plot 4: Inflation vs Belt Tightening vs Morale
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    scatter4 = ax4.scatter(valid_results['inflation'], 
                          valid_results['belt_tightening'], 
                          valid_results['final_average_morale'],
                          c=valid_results['final_average_morale'], 
                          cmap='RdYlGn', alpha=0.6)
    ax4.set_xlabel('Inflation')
    ax4.set_ylabel('Belt Tightening')
    ax4.set_zlabel('Final Average Morale')
    ax4.set_title('Inflation vs Belt Tightening vs Morale')
    plt.colorbar(scatter4, ax=ax4, shrink=0.5)
    
    # Plot 5: Private Morale Alpha vs Sigmoid Gamma vs Striking Workers
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    scatter5 = ax5.scatter(valid_results['private_morale_alpha'], 
                          valid_results['sigmoid_gamma'], 
                          valid_results['final_striking_workers'],
                          c=valid_results['final_striking_workers'], 
                          cmap='coolwarm', alpha=0.6)
    ax5.set_xlabel('Private Morale Alpha')
    ax5.set_ylabel('Sigmoid Gamma')
    ax5.set_zlabel('Final Striking Workers')
    ax5.set_title('Private Morale Alpha vs Sigmoid Gamma vs Striking Workers')
    plt.colorbar(scatter5, ax=ax5, shrink=0.5)
    
    # Plot 6: All parameters vs Striking Workers (using color for 4th parameter)
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    scatter6 = ax6.scatter(valid_results['inflation'], 
                          valid_results['belt_tightening'], 
                          valid_results['final_striking_workers'],
                          c=valid_results['sigmoid_gamma'], 
                          cmap='viridis', alpha=0.6)
    ax6.set_xlabel('Inflation')
    ax6.set_ylabel('Belt Tightening')
    ax6.set_zlabel('Final Striking Workers')
    ax6.set_title('Inflation vs Belt Tightening vs Striking Workers\n(Color: Sigmoid Gamma)')
    plt.colorbar(scatter6, ax=ax6, shrink=0.5, label='Sigmoid Gamma')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3d_parameter_visualizations.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_visualizations(results_df, output_dir):
    """Create summary visualizations of the parameter sweep results"""
    
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
    fig.suptitle('Parameter Sweep Summary', fontsize=16, fontweight='bold')
    
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
                         c=valid_results['inflation'], cmap='viridis', alpha=0.6)
    ax3.set_title('Striking Workers vs Average Morale\n(Color: Inflation)')
    ax3.set_xlabel('Final Striking Workers')
    ax3.set_ylabel('Final Average Morale')
    plt.colorbar(scatter, ax=ax3, label='Inflation')
    
    # Plot 4: Correlation heatmap of all parameters
    ax4 = axes[1, 1]
    param_cols = ['inflation', 'belt_tightening', 'sigmoid_gamma', 'private_morale_alpha']
    metric_cols = ['final_striking_workers', 'final_average_morale']
    correlation_data = valid_results[param_cols + metric_cols].corr()
    
    sns.heatmap(correlation_data, annot=True, cmap='RdBu_r', center=0, 
                square=True, ax=ax4, fmt='.3f')
    ax4.set_title('Parameter Correlations')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_sweep_summary.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def run_parameter_sweep():
    """Run parameter sweep with specified ranges"""
    
    # Load base settings
    base_settings = load_emory_settings()
    
    # Define parameter ranges
    inflation_range = np.arange(0.01, 0.21, 0.1)  # 0 to 0.2 in 0.1 steps
    belt_tightening_range = np.arange(-0.7, 0.71, 0.1)  # -1 to 1 in 0.1 steps
    sigmoid_gamma_range = np.arange(0.0, 1.01, 0.1)  # 0 to 1 in 0.1 steps
    private_morale_alpha_range = np.arange(0.5, 1.01, 0.1)  # 0 to 1 in 0.1 steps
    
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
                            'max_striking_workers': max(sim_data['striking_workers']) if sim_data['striking_workers'] else 0,
                            'min_striking_workers': min(sim_data['striking_workers']) if sim_data['striking_workers'] else 0,
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
    # Run the parameter sweep
    results, output_directory = run_parameter_sweep()
    print(f"\nParameter sweep completed successfully!")
    print(f"Check the output directory '{output_directory}' for all results and visualizations.")
