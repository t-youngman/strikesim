#!/usr/bin/env python3
"""
Simple test to check robustness issues
"""

import sys
import os
sys.path.append('.')

from strikesim import StrikeSimulation

def test_basic_robustness():
    """Test basic robustness issues"""
    
    print("Testing basic robustness...")
    
    # Test 1: Very small network
    try:
        settings = {
            'start_date': '2024-01-01',
            'duration': 5,
            'working_days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
            'num_workers': 3  # Very small
        }
        sim = StrikeSimulation(settings)
        sim.initialize_simulation()
        print(f"✓ Small network (3 workers): {len(sim.workers)} workers created")
    except Exception as e:
        print(f"❌ Small network failed: {e}")
    
    # Test 2: Large network
    try:
        settings = {
            'start_date': '2024-01-01',
            'duration': 5,
            'working_days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
            'num_workers': 100  # Large
        }
        sim = StrikeSimulation(settings)
        sim.initialize_simulation()
        print(f"✓ Large network (100 workers): {len(sim.workers)} workers created")
    except Exception as e:
        print(f"❌ Large network failed: {e}")
    
    # Test 3: Check for potential issues with network size mismatch
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
        
        print(f"✓ Network sizes: Employer={employer_nodes}, Union={union_nodes}, Workers={worker_count}")
        
        if employer_nodes != worker_count:
            print(f"⚠️  Warning: Employer network size ({employer_nodes}) != worker count ({worker_count})")
        
    except Exception as e:
        print(f"❌ Network size test failed: {e}")

if __name__ == "__main__":
    test_basic_robustness() 