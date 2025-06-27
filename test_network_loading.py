#!/usr/bin/env python3
"""
Test script to verify network loading functionality.
"""

import settings
from strikesim import StrikeSimulation
import os

def test_network_loading():
    """Test the network loading functionality"""
    
    # Convert settings to dictionary
    settings_dict = {key: value for key, value in vars(settings).items() 
                    if not key.startswith('_')}
    
    print("Testing network loading functionality...")
    print()
    
    # Test 1: Check available networks
    sim = StrikeSimulation(settings_dict)
    available_networks = sim.get_available_networks()
    print("Available networks:")
    print(f"  Employer networks: {available_networks['employers']}")
    print(f"  Union networks: {available_networks['unions']}")
    print()
    
    # Test 2: Test with generated networks (default)
    print("Test 1: Generated networks (default)")
    settings_dict['employer_network_file'] = None
    settings_dict['union_network_file'] = None
    sim1 = StrikeSimulation(settings_dict)
    sim1.initialize_simulation()  # Initialize the simulation
    print(f"  Employer network: {sim1.employer_network.number_of_nodes()} nodes, {sim1.employer_network.number_of_edges()} edges")
    print(f"  Union network: {sim1.union_network.number_of_nodes()} nodes, {sim1.union_network.number_of_edges()} edges")
    print()
    
    # Test 3: Test with loaded employer network (if available)
    if available_networks['employers']:
        print("Test 2: Loaded employer network")
        settings_dict['employer_network_file'] = available_networks['employers'][0]
        settings_dict['union_network_file'] = None
        sim2 = StrikeSimulation(settings_dict)
        sim2.initialize_simulation()  # Initialize the simulation
        print(f"  Employer network: {sim2.employer_network.number_of_nodes()} nodes, {sim2.employer_network.number_of_edges()} edges")
        print(f"  Union network: {sim2.union_network.number_of_nodes()} nodes, {sim2.union_network.number_of_edges()} edges")
        print()
    
    # Test 4: Test with non-existent file (should fallback to generated)
    print("Test 3: Non-existent network file (should fallback to generated)")
    settings_dict['employer_network_file'] = 'non_existent_file'
    settings_dict['union_network_file'] = None
    sim3 = StrikeSimulation(settings_dict)
    sim3.initialize_simulation()  # Initialize the simulation
    print(f"  Employer network: {sim3.employer_network.number_of_nodes()} nodes, {sim3.employer_network.number_of_edges()} edges")
    print(f"  Union network: {sim3.union_network.number_of_nodes()} nodes, {sim3.union_network.number_of_edges()} edges")
    print()
    
    # Test 5: Test with both networks loaded (if available)
    if available_networks['employers'] and available_networks['unions']:
        print("Test 4: Both networks loaded")
        settings_dict['employer_network_file'] = available_networks['employers'][0]
        settings_dict['union_network_file'] = available_networks['unions'][0]
        sim4 = StrikeSimulation(settings_dict)
        sim4.initialize_simulation()  # Initialize the simulation
        print(f"  Employer network: {sim4.employer_network.number_of_nodes()} nodes, {sim4.employer_network.number_of_edges()} edges")
        print(f"  Union network: {sim4.union_network.number_of_nodes()} nodes, {sim4.union_network.number_of_edges()} edges")
        print()
    
    print("Network loading tests completed successfully!")

if __name__ == "__main__":
    test_network_loading() 