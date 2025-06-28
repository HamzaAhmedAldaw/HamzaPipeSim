#!/usr/bin/env python3
"""
Working Elevation Profile Example
Uses pressure boundary conditions at both ends for guaranteed convergence
"""

import sys
import os
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Load Pipeline-Sim
print("Loading Pipeline-Sim...")
pyd_path = r"C:\Users\KIMO STORE\AppData\Roaming\Python\Python313\site-packages\pipeline_sim.cp313-win_amd64.pyd"
spec = importlib.util.spec_from_file_location("pipeline_sim", pyd_path)
pipeline_sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline_sim)

# Import classes
Network = pipeline_sim.Network
NodeType = pipeline_sim.NodeType
FluidProperties = pipeline_sim.FluidProperties
SteadyStateSolver = pipeline_sim.SteadyStateSolver
constants = pipeline_sim.constants

print("✅ Pipeline-Sim loaded successfully")

def create_simple_elevation_profile():
    """Create a simple pipeline with elevation changes"""
    
    print("\n=== Creating Pipeline with Elevation Profile ===")
    
    # Create network with just inlet and outlet
    network = Network()
    
    # Platform at sea level
    platform = network.add_node("platform", NodeType.SOURCE)
    
    # Terminal at higher elevation
    terminal = network.add_node("terminal", NodeType.SINK)
    
    # Single pipe: 10 km long, 0.5 m diameter
    pipe = network.add_pipe("pipeline", platform, terminal, 10000.0, 0.5)
    
    # Set pressures (both ends need pressure for solver to work)
    network.set_pressure(platform, 80e5)   # 80 bar at platform
    network.set_pressure(terminal, 30e5)   # 30 bar at terminal
    
    # Note: We'll simulate elevation effect in the visualization
    # The solver doesn't use node elevation property
    platform_elevation = 0     # Sea level
    terminal_elevation = 100   # 100 m above sea level
    
    return network, platform, terminal, pipe, platform_elevation, terminal_elevation

def solve_pipeline(network, elevation_change):
    """Solve the pipeline"""
    
    # Heavy crude oil
    fluid = FluidProperties()
    fluid.oil_fraction = 1.0
    fluid.oil_density = 920.0      # kg/m³
    fluid.oil_viscosity = 0.030    # Pa.s (30 cP)
    
    print(f"\nFluid properties:")
    print(f"  Density: {fluid.oil_density} kg/m³")
    print(f"  Viscosity: {fluid.oil_viscosity*1000:.0f} cP")
    
    # Calculate static head
    static_head_pa = fluid.oil_density * constants.GRAVITY * elevation_change
    static_head_bar = static_head_pa / 1e5
    print(f"\nStatic head due to {elevation_change} m elevation:")
    print(f"  {static_head_bar:.2f} bar ({static_head_pa:.0f} Pa)")
    
    # Create solver
    solver = SteadyStateSolver(network, fluid)
    config = solver.config
    config.verbose = True
    config.tolerance = 1e-6
    solver.set_config(config)
    
    print("\nSolving pipeline...")
    results = solver.solve()
    
    return results, fluid, static_head_bar

def create_multi_segment_elevation():
    """Create multi-segment pipeline with elevation profile"""
    
    print("\n=== Creating Multi-Segment Pipeline ===")
    
    # Profile points
    profile = [
        ("inlet", 0, 0),       # Start at ground level
        ("hill_1", 3, 50),     # First hill
        ("valley", 6, 20),     # Valley
        ("hill_2", 9, 80),     # Second hill
        ("outlet", 12, 40),    # End point
    ]
    
    # Create network
    network = Network()
    nodes = []
    
    # Create nodes
    for i, (name, _, _) in enumerate(profile):
        if i == 0:
            node = network.add_node(name, NodeType.SOURCE)
        elif i == len(profile) - 1:
            node = network.add_node(name, NodeType.SINK)
        else:
            node = network.add_node(name, NodeType.JUNCTION)
        nodes.append(node)
    
    # Create pipes
    pipes = []
    for i in range(len(nodes) - 1):
        dist = (profile[i+1][1] - profile[i][1]) * 1000  # Convert km to m
        pipe = network.add_pipe(f"segment_{i+1}", nodes[i], nodes[i+1], dist, 0.4)
        pipes.append(pipe)
    
    # Set boundary conditions
    network.set_pressure(nodes[0], 70e5)    # 70 bar at inlet
    network.set_pressure(nodes[-1], 25e5)   # 25 bar at outlet
    
    return network, nodes, pipes, profile

def plot_elevation_results(results, fluid, static_head_bar, elev_start, elev_end):
    """Plot results for simple elevation pipeline"""
    
    if not results or not results.converged:
        print("❌ Cannot plot - solution did not converge")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Get results
    p_start = results.node_pressures["platform"] / 1e5
    p_end = results.node_pressures["terminal"] / 1e5
    flow = list(results.pipe_flow_rates.values())[0]
    velocity = list(results.pipe_velocities.values())[0]
    
    # Total pressure drop
    total_dp = p_start - p_end
    friction_dp = total_dp - static_head_bar
    
    print(f"\nResults Summary:")
    print(f"  Flow rate: {flow:.3f} m³/s ({flow*3600:.1f} m³/h)")
    print(f"  Velocity: {velocity:.2f} m/s")
    print(f"  Total pressure drop: {total_dp:.2f} bar")
    print(f"  Friction component: {friction_dp:.2f} bar")
    print(f"  Static head component: {static_head_bar:.2f} bar")
    
    # 1. Elevation and pressure profile
    distances = [0, 10]  # km
    elevations = [elev_start, elev_end]
    pressures = [p_start, p_end]
    
    # Plot elevation
    ax1.fill_between(distances, 0, elevations, alpha=0.3, color='brown', label='Terrain')
    ax1.plot(distances, elevations, 'k-', linewidth=2, label='Pipeline Path')
    
    # Plot pressure on secondary axis
    ax1_p = ax1.twinx()
    ax1_p.plot(distances, pressures, 'b-o', linewidth=2, markersize=10, label='Pressure')
    
    # Draw pipeline
    ax1.plot(distances, elevations, 'gray', linewidth=8, alpha=0.5)
    
    ax1.set_xlabel('Distance (km)', fontsize=12)
    ax1.set_ylabel('Elevation (m)', fontsize=12)
    ax1_p.set_ylabel('Pressure (bar)', fontsize=12, color='b')
    ax1.set_title('Pipeline Elevation and Pressure Profile', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_p.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # 2. Pressure components
    components = ['Friction Loss', 'Static Head', 'Total']
    values = [friction_dp, static_head_bar, total_dp]
    colors = ['red', 'blue', 'purple']
    
    bars = ax2.bar(components, values, color=colors, alpha=0.7)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:.1f} bar', ha='center', va='bottom', fontsize=10)
    
    ax2.set_ylabel('Pressure Drop (bar)', fontsize=12)
    ax2.set_title('Pressure Drop Components', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    for bar, val in zip(bars[:2], [friction_dp, static_head_bar]):
        percent = (val / total_dp) * 100
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height()/2,
                f'{percent:.0f}%', ha='center', va='center', 
                fontsize=12, color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def plot_multi_segment_results(network, nodes, profile, results, fluid):
    """Plot multi-segment elevation profile"""
    
    if not results or not results.converged:
        print("❌ Cannot plot - solution did not converge")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Extract data
    distances = [point[1] for point in profile]
    elevations = [point[2] for point in profile]
    pressures = [results.node_pressures[node.id] / 1e5 for node in nodes]
    
    # 1. Elevation and pressure profile
    ax1_p = ax1.twinx()
    
    # Elevation
    ax1.fill_between(distances, 0, elevations, alpha=0.3, color='brown')
    ax1.plot(distances, elevations, 'k-', linewidth=2, label='Elevation')
    
    # Add markers for nodes
    for i, (name, dist, elev) in enumerate(profile):
        ax1.plot(dist, elev, 'ko', markersize=8)
        ax1.annotate(name, (dist, elev), xytext=(dist, elev + 10),
                    ha='center', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Pressure
    ax1_p.plot(distances, pressures, 'b-o', linewidth=2, markersize=8, label='Pressure')
    
    # Pressure drop annotations
    for i in range(len(pressures) - 1):
        dp = pressures[i] - pressures[i+1]
        mid_dist = (distances[i] + distances[i+1]) / 2
        mid_press = (pressures[i] + pressures[i+1]) / 2
        ax1_p.annotate(f'ΔP={dp:.1f} bar', 
                      xy=(mid_dist, mid_press),
                      xytext=(mid_dist, mid_press + 3),
                      ha='center', fontsize=9,
                      bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.7),
                      arrowprops=dict(arrowstyle='->', color='blue', lw=1))
    
    ax1.set_ylabel('Elevation (m)', fontsize=12)
    ax1_p.set_ylabel('Pressure (bar)', fontsize=12, color='b')
    ax1.set_title('Multi-Segment Pipeline: Elevation and Pressure Profiles', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1_p.tick_params(axis='y', labelcolor='b')
    
    # 2. Flow parameters
    if len(results.pipe_velocities) > 0:
        segment_centers = [(distances[i] + distances[i+1])/2 for i in range(len(distances)-1)]
        velocities = [results.pipe_velocities.get(f"segment_{i+1}", 0) for i in range(len(nodes)-1)]
        
        ax2.bar(segment_centers, velocities, width=0.8, alpha=0.7, color='green')
        ax2.set_xlabel('Distance (km)', fontsize=12)
        ax2.set_ylabel('Velocity (m/s)', fontsize=12)
        ax2.set_title('Flow Velocity in Each Segment', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add flow rate annotation
        flow = list(results.pipe_flow_rates.values())[0] if results.pipe_flow_rates else 0
        ax2.text(0.02, 0.95, f'Flow rate: {flow:.3f} m³/s ({flow*3600:.1f} m³/h)',
                transform=ax2.transAxes, fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    plt.tight_layout()
    plt.show()

def main():
    """Run elevation profile examples"""
    
    print("\n" + "="*60)
    print("Pipeline-Sim: Elevation Profile Analysis")
    print("="*60)
    
    try:
        # Example 1: Simple pipeline with elevation
        print("\n--- Example 1: Simple Pipeline with Elevation ---")
        network1, platform, terminal, pipe, elev_start, elev_end = create_simple_elevation_profile()
        elevation_change = elev_end - elev_start
        
        results1, fluid1, static_head1 = solve_pipeline(network1, elevation_change)
        
        if results1.converged:
            print("\n✅ Solution converged!")
            plot_elevation_results(results1, fluid1, static_head1, elev_start, elev_end)
        else:
            print("\n❌ Solution did not converge")
        
        # Example 2: Multi-segment with elevation profile
        print("\n--- Example 2: Multi-Segment Pipeline ---")
        network2, nodes2, pipes2, profile2 = create_multi_segment_elevation()
        
        # Solve with lighter fluid for better convergence
        fluid2 = FluidProperties()
        fluid2.oil_density = 850.0
        fluid2.oil_viscosity = 0.010
        
        solver2 = SteadyStateSolver(network2, fluid2)
        config2 = solver2.config
        config2.verbose = True
        solver2.set_config(config2)
        
        print("\nSolving multi-segment pipeline...")
        results2 = solver2.solve()
        
        if results2.converged:
            print("\n✅ Solution converged!")
            plot_multi_segment_results(network2, nodes2, profile2, results2, fluid2)
        else:
            print("\n❌ Solution did not converge")
        
        print("\n" + "="*60)
        print("✅ Analysis completed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()