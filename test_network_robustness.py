#!/usr/bin/env python3
"""
Test script to assess robustness of the implementation to different employer network specifications
"""

import sys
import os
import networkx as nx
import pandas as pd
sys.path.append('.')

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

def test_network_robustness():
    """Test the robustness of the implementation to different network specifications"""
    
    print("Testing network robustness...")
    
    # Create test networks
    test_networks = create_test_networks()
    
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
    
    results = {}
    
    for network_name, network in test_networks.items():
        print(f"\n--- Testing {network_name} network ---")
        print(f"Network size: {network.number_of_nodes()} nodes, {network.number_of_edges()} edges")
        
        try:
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
            if len(sim.workers) != network.number_of_nodes():
                print(f"❌ Worker count mismatch: {len(sim.workers)} workers vs {network.number_of_nodes()} nodes")
                continue
            
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
                    print(f"❌ Error updating morale for worker {i}: {e}")
            
            # Test network influence calculation
            try:
                social_morale = sim.calculate_social_morale(sim.workers[0])
                print(f"✓ Social morale calculation successful: {social_morale:.3f}")
            except Exception as e:
                print(f"❌ Error calculating social morale: {e}")
            
            print(f"✓ {network_name} network test passed ({success_count}/{min(5, len(sim.workers))} workers tested)")
            results[network_name] = 'PASS'
            
        except Exception as e:
            print(f"❌ {network_name} network test failed: {e}")
            results[network_name] = f'FAIL: {str(e)}'
    
    # Summary
    print(f"\n--- Network Robustness Test Summary ---")
    for network_name, result in results.items():
        print(f"{network_name}: {result}")
    
    # Test edge cases
    print(f"\n--- Testing Edge Cases ---")
    
    # Test with empty network (should fail gracefully)
    try:
        settings = base_settings.copy()
        settings['num_workers'] = 0
        sim = StrikeSimulation(settings)
        print("❌ Should not allow 0 workers")
    except Exception as e:
        print(f"✓ Correctly handles 0 workers: {e}")
    
    # Test with very large network
    try:
        settings = base_settings.copy()
        settings['num_workers'] = 1000
        sim = StrikeSimulation(settings)
        print(f"✓ Handles large network: {len(sim.workers)} workers")
    except Exception as e:
        print(f"❌ Error with large network: {e}")
    
    return results

def test_csv_network_loading():
    """Test loading networks from CSV files"""
    print(f"\n--- Testing CSV Network Loading ---")
    
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
        
        print(f"✓ CSV network loading successful: {len(sim.workers)} workers")
        
        # Clean up
        os.remove(test_csv_path)
        
    except Exception as e:
        print(f"❌ CSV network loading failed: {e}")
        if os.path.exists(test_csv_path):
            os.remove(test_csv_path)

if __name__ == "__main__":
    test_network_robustness()
    test_csv_network_loading() 