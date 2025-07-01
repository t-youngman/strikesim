#!/usr/bin/env python3
"""
Basic functionality tests for StrikeSim
"""

import sys
import os
import pytest

# Add parent directory to path to import strikesim
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strikesim import StrikeSimulation
from datetime import datetime

def test_basic_simulation():
    """Test basic simulation functionality"""
    
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
    
    sim = StrikeSimulation(settings)
    results = sim.run_simulation()
    
    # Check that results contain expected keys
    expected_keys = ['dates', 'striking_workers', 'working_workers', 'employer_balance', 
                    'union_balance', 'average_morale', 'average_savings', 'worker_states']
    
    for key in expected_keys:
        assert key in results, f"Results should contain '{key}'"
    
    # Check that results have the expected length
    assert len(results['dates']) > 0, "Simulation should have at least one day"
    assert len(results['striking_workers']) == len(results['dates']), "All arrays should have same length"
    assert len(results['working_workers']) == len(results['dates']), "All arrays should have same length"

def test_network_generation():
    """Test network generation functionality"""
    
    settings = {
        'start_date': '2024-01-01',
        'duration': 5,
        'working_days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'num_workers': 15
    }
    
    sim = StrikeSimulation(settings)
    sim.initialize_simulation()
    
    # Check that networks are generated
    assert sim.employer_network is not None, "Employer network should be generated"
    assert sim.union_network is not None, "Union network should be generated"
    
    # Check network sizes
    assert sim.employer_network.number_of_nodes() == 15, "Employer network should have 15 nodes"
    assert sim.union_network.number_of_nodes() > 0, "Union network should have nodes"
    
    # Check that workers are created
    assert len(sim.workers) == 15, "Should have 15 workers"

def test_morale_calculation():
    """Test morale calculation functionality"""
    
    settings = {
        'start_date': '2024-01-01',
        'duration': 5,
        'working_days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'num_workers': 10,
        'initial_wage': 100.0,
        'target_wage': 105.0,
        'initial_employer_balance': 10000.0,
        'initial_strike_fund': 5000.0
    }
    
    sim = StrikeSimulation(settings)
    sim.initialize_simulation()
    
    # Test morale calculation for a worker
    worker = sim.workers[0]
    initial_morale = worker.morale
    
    sim.update_worker_morale(worker)
    new_morale = worker.morale
    
    # Morale should be a number between 0 and 1
    assert isinstance(new_morale, (int, float)), "Morale should be a number"
    assert 0 <= new_morale <= 1, "Morale should be between 0 and 1"

def test_strike_day_detection():
    """Test strike day detection functionality"""
    
    settings = {
        'start_date': '2024-01-01',
        'duration': 10,
        'working_days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'strike_pattern': 'once_a_week'
    }
    
    sim = StrikeSimulation(settings)
    
    # Test specific dates
    monday = datetime(2024, 1, 1)  # Monday
    tuesday = datetime(2024, 1, 2)  # Tuesday
    
    assert sim.is_strike_day(monday), "Monday should be a strike day"
    assert not sim.is_strike_day(tuesday), "Tuesday should not be a strike day"

def test_worker_states():
    """Test worker state management"""
    
    settings = {
        'start_date': '2024-01-01',
        'duration': 5,
        'working_days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'num_workers': 10,
        'initial_wage': 100.0,
        'target_wage': 105.0,
        'initial_employer_balance': 10000.0,
        'initial_strike_fund': 5000.0
    }
    
    sim = StrikeSimulation(settings)
    sim.initialize_simulation()
    
    # Check initial worker states
    for worker in sim.workers:
        assert worker.state in ['striking', 'not_striking'], f"Worker state should be valid, got {worker.state}"
    
    # Test state changes
    worker = sim.workers[0]
    original_state = worker.state
    
    # Change state
    worker.state = 'striking' if original_state == 'not_striking' else 'not_striking'
    assert worker.state != original_state, "Worker state should change"

def test_financial_flows():
    """Test financial flow calculations"""
    
    settings = {
        'start_date': '2024-01-01',
        'duration': 5,
        'working_days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'num_workers': 10,
        'initial_wage': 100.0,
        'target_wage': 105.0,
        'initial_employer_balance': 10000.0,
        'initial_strike_fund': 5000.0
    }
    
    sim = StrikeSimulation(settings)
    sim.initialize_simulation()
    
    # Test financial flows
    initial_employer_balance = sim.employer.balance
    initial_union_balance = sim.union.strike_fund_balance
    
    sim.process_daily_financial_flows()
    
    # Balances should change
    assert sim.employer.balance != initial_employer_balance, "Employer balance should change"
    assert sim.union.strike_fund_balance != initial_union_balance, "Union balance should change" 