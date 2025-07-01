#!/usr/bin/env python3
"""
Test script to assess robustness of the implementation to different employer network specifications
"""

import sys
import os
import networkx as nx
import pandas as pd
import pytest

# Add parent directory to path to import strikesim
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strikesim import StrikeSimulation
from datetime import datetime

def create_test_networks():
    """Create various test networks to test robustness"""
    networks = {}
    
    # Test 1: Small network (5 workers)
    G_small = nx.Graph()
    for i in range(5):
        G_small.add_node(i, level='worker', type='worker', is_manager=(i < 2))
    G_small.add_edge(0, 1)  # Connect managers
    G_small.add_edge(1, 2)  # Connect manager to worker
    G_small.add_edge(2, 3)  # Connect workers
    G_small.add_edge(3, 4)  # Connect workers
    networks['small'] = G_small
    
    # Test 2: Large network (50 workers)
    G_large = nx.Graph()
    for i in range(50):
        G_large.add_node(i, level='worker', type='worker', is_manager=(i < 10))
    # Create some connections
    for i in range(49):
        G_large.add_edge(i, i+1)
    networks['large'] = G_large
    
    # Test 3: Disconnected network
    G_disconnected = nx.Graph()
    for i in range(10):
        G_disconnected.add_node(i, level='worker', type='worker', is_manager=(i < 3))
    # Create two disconnected components
    G_disconnected.add_edge(0, 1)
    G_disconnected.add_edge(1, 2)
    G_disconnected.add_edge(5, 6)
    G_disconnected.add_edge(6, 7)
    networks['disconnected'] = G_disconnected
    
    # Test 4: Dense network
    G_dense = nx.Graph()
    for i in range(8):
        G_dense.add_node(i, level='worker', type='worker', is_manager=(i < 3))
    # Connect everyone to everyone
    for i in range(8):
        for j in range(i+1, 8):
            G_dense.add_edge(i, j)
    networks['dense'] = G_dense
    
    # Test 5: Star network (centralized)
    G_star = nx.Graph()
    for i in range(10):
        G_star.add_node(i, level='worker', type='worker', is_manager=(i == 0))
    # Connect everyone to center
    for i in range(1, 10):
        G_star.add_edge(0, i)
    networks['star'] = G_star
    
    return networks

@pytest.mark.parametrize("network_name", ["small", "large", "disconnected", "dense", "star"])
def test_network_robustness(network_name):
    """Test the robustness of the implementation to different network specifications"""
    
    # Create test networks
    test_networks = create_test_networks()
    network = test_networks[network_name]
    
    # Base settings
    base_settings = {
        'start_date': '2024-01-01',
        'duration': 5,
        'working_days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'strike_pattern': 'once_a_week',
        'initial_wage': 100.0,
        'target_wage': 105.0,
        'initial_employer_balance': 10000.0,
        'initial_strike_fund': 5000.0,
        'morale_specification': 'sigmoid',
        'private_morale_alpha': 0.8,
        'social_morale_beta': 0.2
    }
    
    # Create settings for this network
    settings = base_settings.copy()
    settings['num_workers'] = network.number_of_nodes()
    
    # Create simulation
    sim = StrikeSimulation(settings)
    
    # Replace the generated employer network with our test network
    sim.employer_network = network
    
    # Initialize simulation
    sim.initialize_simulation()
    
    # Verify worker count matches network size
    assert len(sim.workers) == network.number_of_nodes(), f"Worker count mismatch: {len(sim.workers)} workers vs {network.number_of_nodes()} nodes"
    
    # Test basic functionality
    sim.current_date = datetime(2024, 1, 1)  # Monday - strike day
    
    # Set some workers to striking
    for i in range(min(3, len(sim.workers))):
        sim.workers[i].state = 'striking'
        sim.workers[i].morale = 0.8
    
    # Test morale calculation
    success_count = 0
    for i in range(min(5, len(sim.workers))):
        try:
            old_morale = sim.workers[i].morale
            sim.update_worker_morale(sim.workers[i])
            new_morale = sim.workers[i].morale
            success_count += 1
        except Exception as e:
            pytest.fail(f"Error updating morale for worker {i}: {e}")
    
    # Test network influence calculation
    try:
        social_morale = sim.calculate_social_morale(sim.workers[0])
        assert isinstance(social_morale, (int, float)), "Social morale should be a number"
    except Exception as e:
        pytest.fail(f"Error calculating social morale: {e}")
    
    assert success_count > 0, f"No workers could be tested for {network_name} network"

def test_csv_network_loading():
    """Test loading networks from CSV files"""
    
    # Create a test CSV file
    test_edges = pd.DataFrame({
        'From': [0, 1, 2, 3, 4],
        'To': [1, 2, 3, 4, 0]
    })
    
    # Save test CSV
    test_csv_path = 'networks/employers/test_robustness.csv'
    os.makedirs('networks/employers', exist_ok=True)
    test_edges.to_csv(test_csv_path, index=False)
    
    try:
        settings = {
            'start_date': '2024-01-01',
            'duration': 5,
            'working_days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
            'employer_network_file': 'test_robustness'
        }
        
        sim = StrikeSimulation(settings)
        sim.initialize_simulation()
        
        assert len(sim.workers) == 5, f"Expected 5 workers, got {len(sim.workers)}"
        
    except Exception as e:
        pytest.fail(f"CSV network loading failed: {e}")
    finally:
        # Clean up
        if os.path.exists(test_csv_path):
            os.remove(test_csv_path)

def test_edge_cases():
    """Test edge cases for network robustness"""
    
    # Test with very small network
    try:
        settings = {
            'start_date': '2024-01-01',
            'duration': 5,
            'working_days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
            'num_workers': 3  # Very small
        }
        sim = StrikeSimulation(settings)
        sim.initialize_simulation()
        assert len(sim.workers) == 3, f"Expected 3 workers, got {len(sim.workers)}"
    except Exception as e:
        pytest.fail(f"Small network failed: {e}")
    
    # Test with large network
    try:
        settings = {
            'start_date': '2024-01-01',
            'duration': 5,
            'working_days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
            'num_workers': 100  # Large
        }
        sim = StrikeSimulation(settings)
        sim.initialize_simulation()
        assert len(sim.workers) == 100, f"Expected 100 workers, got {len(sim.workers)}"
    except Exception as e:
        pytest.fail(f"Large network failed: {e}")
    
    # Test network size consistency
    try:
        settings = {
            'start_date': '2024-01-01',
            'duration': 5,
            'working_days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
            'num_workers': 10
        }
        sim = StrikeSimulation(settings)
        sim.initialize_simulation()
        
        # Check if worker count matches network size
        employer_nodes = sim.employer_network.number_of_nodes()
        union_nodes = sim.union_network.number_of_nodes()
        worker_count = len(sim.workers)
        
        assert employer_nodes == worker_count, f"Employer network size ({employer_nodes}) != worker count ({worker_count})"
        
    except Exception as e:
        pytest.fail(f"Network size test failed: {e}") 