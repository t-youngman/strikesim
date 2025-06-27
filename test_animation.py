#!/usr/bin/env python3
"""
Test script for StrikeSim network animation functionality
"""

from strikesim import StrikeSimulation
import settings

def test_animation():
    """Test the network animation functionality"""
    print("Testing StrikeSim Network Animation...")
    
    # Create a simple simulation
    sim = StrikeSimulation(settings.__dict__)
    results = sim.run_simulation()
    
    print(f"Simulation completed with {len(results['striking_workers'])} days")
    print(f"Final outcome: {sim.analyze_results()['outcome']}")
    
    # Create animation
    print("\nCreating network animation...")
    animation_path = sim.create_network_animation(
        save_path='test_animation.gif',
        fps=2,
        max_frames=20  # Limit frames for testing
    )
    
    if animation_path:
        print(f"✅ Animation created successfully: {animation_path}")
        print("You can now view the animation in your file browser!")
    else:
        print("❌ Failed to create animation")

if __name__ == "__main__":
    test_animation() 