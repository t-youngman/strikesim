#!/usr/bin/env python3
"""
Test script to verify network visualization functionality
"""

import sys
import os
import pytest
import matplotlib.pyplot as plt

# Add parent directory to path to import strikesim
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strikesim import StrikeSimulation
from datetime import datetime

def test_network_visualization():
    """Test that network visualization works correctly"""
    
    # Create a simple simulation
    settings = {
        'start_date': '2024-01-01',
        'duration': 10,
        'working_days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'strike_pattern': 'once_a_week',
        'num_workers': 10,
        'initial_wage': 100.0,
        'target_wage': 105.0,
        'initial_employer_balance': 10000.0,
        'initial_strike_fund': 5000.0
    }
    
    # Create simulation
    sim = StrikeSimulation(settings)
    results = sim.run_simulation()
    
    assert len(results['worker_states']) > 0, "Simulation should have worker states"
    
    # Test visualization at first timestep
    fig_first = sim.visualize_networks_at_timestep(0)
    assert fig_first is not None, "First timestep visualization should not be None"
    plt.close(fig_first)
    
    # Test visualization at final timestep
    final_timestep = len(results['worker_states']) - 1
    fig_final = sim.visualize_networks_at_timestep(final_timestep)
    assert fig_final is not None, "Final timestep visualization should not be None"
    plt.close(fig_final)
    
    # Test with invalid timestep
    fig_invalid = sim.visualize_networks_at_timestep(999)
    assert fig_invalid is None, "Invalid timestep should return None"

def test_visualization_with_different_network_sizes():
    """Test visualization with different network sizes"""
    
    for num_workers in [5, 10, 20]:
        settings = {
            'start_date': '2024-01-01',
            'duration': 5,
            'working_days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
            'strike_pattern': 'once_a_week',
            'num_workers': num_workers,
            'initial_wage': 100.0,
            'target_wage': 105.0,
            'initial_employer_balance': 10000.0,
            'initial_strike_fund': 5000.0
        }
        
        sim = StrikeSimulation(settings)
        results = sim.run_simulation()
        
        # Test visualization works for this network size
        fig = sim.visualize_networks_at_timestep(0)
        assert fig is not None, f"Visualization should work for {num_workers} workers"
        plt.close(fig)

def test_visualization_with_striking_workers():
    """Test visualization when workers are striking"""
    
    settings = {
        'start_date': '2024-01-01',
        'duration': 5,
        'working_days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'strike_pattern': 'once_a_week',
        'num_workers': 10,
        'initial_wage': 100.0,
        'target_wage': 105.0,
        'initial_employer_balance': 10000.0,
        'initial_strike_fund': 5000.0
    }
    
    sim = StrikeSimulation(settings)
    results = sim.run_simulation()
    
    # Find a timestep where some workers are striking
    striking_timestep = None
    for i, worker_states in enumerate(results['worker_states']):
        if 'striking' in worker_states:
            striking_timestep = i
            break
    
    if striking_timestep is not None:
        fig = sim.visualize_networks_at_timestep(striking_timestep)
        assert fig is not None, "Visualization should work with striking workers"
        plt.close(fig)
    else:
        pytest.skip("No striking workers found in simulation") 