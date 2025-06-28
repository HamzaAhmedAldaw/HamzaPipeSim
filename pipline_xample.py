#!/usr/bin/env python3
"""
Pipeline Profile Analysis with Plotting
Demonstrates pressure, velocity, and elevation profiles along pipelines
"""

import sys
import os
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load Pipeline-Sim
print("Loading Pipeline-Sim...")
pyd_path = r"C:\Users\KIMO STORE\AppData\Roaming\Python\Python313\site-packages\pipeline_sim.cp313-win_amd64.pyd"
spec = importlib.util.spec_from_file_location("pipeline_sim", pyd_path)
pipeline_sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline_sim)

# Import classes
Network = pipeline_sim.Network
Node = pipeline_sim.Node
Pipe = pipeline_sim.Pipe
NodeType = pipeline_sim.NodeType
FluidProperties = pipeline_sim.FluidProperties
SteadyStateSolver = pipeline_sim.SteadyStateSolver
constants = pipeline_sim.constants

def create_pipeline_profile_network():
    """Create a pipeline network with multiple segments for profile analysis"""
    
    network = Network()
    
    # Create nodes along the pipeline
    nodes = []
    node_positions = [0, 2000, 5000, 8000, 10000]  # meters
    node_names = ["inlet", "node1", "node2", "node3", "outlet"]
    
    for i, (name, pos) in enumerate(zip(node_names, node_positions)):
        if i == 0:
            node = network.add_node(name, NodeType.SOURCE)
        elif i == len(node_names) - 1:
            node = network.add_node(name, NodeType.SINK)
        else:
            node = network.add_node(name, NodeType.JUNCTION)
        nodes.append(node)
    
    # Create pipe segments
    pipes = []
    diameters = [0.6, 0.5, 0.5, 0.4]  # Different diameters
    
    for i in range(len(nodes) - 1):
        length = node_positions[i+1] - node_positions[i]
        pipe = network.add_pipe(f"pipe_{i+1}", nodes[i], nodes[i+1], 
                              length, diameters[i])
        pipes.append(pipe)
    
    # Set boundary conditions
    network.set_pressure(nodes[0], 70e5)    # 70 bar at inlet
    network.set_pressure(nodes[-1], 20e5)   # 20 bar at outlet
    
    return network, nodes, pipes, node_positions

def solve_network(network):
    """Solve the network with given fluid properties"""
    
    # Create fluid properties (medium oil)
    fluid = FluidProperties()
    fluid.oil_fraction = 1.0
    fluid.oil_density = 870.0      # kg/m³
    fluid.oil_viscosity = 0.015    # Pa.s (15 cP)
    
    # Create and configure solver
    solver = SteadyStateSolver(network, fluid)
    config = solver.config
    config.verbose = True
    config.tolerance = 1e-6
    solver.set_config(config)
    
    # Solve
    print("\nSolving network...")
    results = solver.solve()
    
    return results, fluid

def plot_pipeline_profiles(network, nodes, pipes, positions, results, fluid):
    """Create comprehensive profile plots"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Get data
    pressures = [results.node_pressures[node.id] / 1e5 for node in nodes]  # Convert to bar
    distances = positions
    
    # Calculate velocities and Reynolds numbers for each pipe
    velocities = []
    reynolds = []
    friction_factors = []
    diameters = []
    
    for pipe_id in results.pipe_flow_rates:
        v = results.pipe_velocities[pipe_id]
        Re = results.pipe_reynolds_numbers[pipe_id]
        f = results.pipe_friction_factors[pipe_id]
        velocities.append(v)
        reynolds.append(Re)
        friction_factors.append(f)
    
    # Get pipe diameters
    for pipe in pipes:
        diameters.append(pipe.diameter)
    
    # 1. Pressure Profile
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(distances, pressures, 'b-o', linewidth=2, markersize=8, label='Pressure')
    ax1.fill_between(distances, pressures, alpha=0.3)
    ax1.set_xlabel('Distance (m)', fontsize=12)
    ax1.set_ylabel('Pressure (bar)', fontsize=12)
    ax1.set_title('Pressure Profile Along Pipeline', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add pressure drop annotations
    for i in range(len(pressures)-1):
        dp = pressures[i] - pressures[i+1]
        mid_x = (distances[i] + distances[i+1]) / 2
        mid_y = (pressures[i] + pressures[i+1]) / 2
        ax1.annotate(f'ΔP = {dp:.1f} bar', 
                    xy=(mid_x, mid_y), 
                    xytext=(mid_x, mid_y + 2),
                    ha='center',
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # 2. Velocity Profile
    ax2 = fig.add_subplot(gs[1, 0])
    pipe_centers = [(distances[i] + distances[i+1])/2 for i in range(len(distances)-1)]
    bars = ax2.bar(pipe_centers, velocities, width=[distances[i+1]-distances[i] 
                   for i in range(len(distances)-1)], alpha=0.7, edgecolor='black')
    
    # Color bars by velocity
    for bar, v in zip(bars, velocities):
        if v < 5:
            bar.set_facecolor('green')
        elif v < 10:
            bar.set_facecolor('orange')
        else:
            bar.set_facecolor('red')
    
    ax2.set_xlabel('Distance (m)', fontsize=12)
    ax2.set_ylabel('Velocity (m/s)', fontsize=12)
    ax2.set_title('Velocity in Each Pipe Segment', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Diameter Profile
    ax3 = fig.add_subplot(gs[1, 1])
    for i in range(len(pipes)):
        ax3.plot([distances[i], distances[i+1]], 
                [diameters[i]*1000, diameters[i]*1000], 
                'g-', linewidth=8, solid_capstyle='butt')
    ax3.set_xlabel('Distance (m)', fontsize=12)
    ax3.set_ylabel('Diameter (mm)', fontsize=12)
    ax3.set_title('Pipe Diameter Profile', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, max(diameters)*1200)
    
    # 4. Reynolds Number
    ax4 = fig.add_subplot(gs[2, 0])
    bars_re = ax4.bar(pipe_centers, reynolds, width=[distances[i+1]-distances[i] 
                      for i in range(len(distances)-1)], alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Distance (m)', fontsize=12)
    ax4.set_ylabel('Reynolds Number', fontsize=12)
    ax4.set_title('Reynolds Number Profile', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=2300, color='r', linestyle='--', label='Laminar/Turbulent Transition')
    ax4.legend()
    
    # 5. Friction Factor
    ax5 = fig.add_subplot(gs[2, 1])
    bars_f = ax5.bar(pipe_centers, friction_factors, width=[distances[i+1]-distances[i] 
                     for i in range(len(distances)-1)], alpha=0.7, edgecolor='black')
    ax5.set_xlabel('Distance (m)', fontsize=12)
    ax5.set_ylabel('Friction Factor', fontsize=12)
    ax5.set_title('Darcy Friction Factor Profile', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Flow Summary
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')
    
    # Calculate total flow
    total_flow = list(results.pipe_flow_rates.values())[0]  # All pipes have same flow
    total_pressure_drop = pressures[0] - pressures[-1]
    
    summary_text = f"""
    FLOW SUMMARY:
    • Total Flow Rate: {total_flow:.3f} m³/s ({total_flow*3600:.1f} m³/h, {total_flow*86400:.0f} m³/day)
    • Total Pressure Drop: {total_pressure_drop:.1f} bar
    • Pressure Gradient: {total_pressure_drop/10:.2f} bar/km
    • Fluid Density: {fluid.oil_density} kg/m³
    • Fluid Viscosity: {fluid.oil_viscosity*1000:.1f} cP
    • Flow Regime: {'Turbulent' if reynolds[0] > 2300 else 'Laminar'}
    """
    
    ax6.text(0.5, 0.5, summary_text, transform=ax6.transAxes,
             fontsize=12, verticalalignment='center', horizontalalignment='center',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.suptitle('Pipeline Hydraulic Profile Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_3d_profile(network, nodes, pipes, positions, results):
    """Create a 3D visualization of the pipeline"""
    
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get pressures
    pressures = [results.node_pressures[node.id] / 1e5 for node in nodes]
    
    # Create 3D pipeline path (add some curvature for visualization)
    x = positions
    y = [0, 100, 200, 100, 0]  # Lateral displacement
    z = [0, 50, 100, 50, 0]    # Elevation changes
    
    # Plot pipeline segments with color based on pressure
    for i in range(len(positions)-1):
        x_seg = [x[i], x[i+1]]
        y_seg = [y[i], y[i+1]]
        z_seg = [z[i], z[i+1]]
        
        # Color based on average pressure
        avg_pressure = (pressures[i] + pressures[i+1]) / 2
        color_intensity = (avg_pressure - min(pressures)) / (max(pressures) - min(pressures))
        
        ax.plot(x_seg, y_seg, z_seg, 
                color=plt.cm.coolwarm(color_intensity), 
                linewidth=8, alpha=0.8)
    
    # Plot nodes
    scatter = ax.scatter(x, y, z, c=pressures, s=200, 
                        cmap='coolwarm', edgecolors='black', linewidth=2)
    
    # Add labels
    for i, (xi, yi, zi, p, node) in enumerate(zip(x, y, z, pressures, nodes)):
        ax.text(xi, yi, zi+20, f'{node.id}\n{p:.1f} bar', 
                fontsize=10, ha='center')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Pressure (bar)', fontsize=12)
    
    ax.set_xlabel('Distance (m)', fontsize=12)
    ax.set_ylabel('Lateral Position (m)', fontsize=12)
    ax.set_zlabel('Elevation (m)', fontsize=12)
    ax.set_title('3D Pipeline Visualization with Pressure', fontsize=14, fontweight='bold')
    
    return fig

def main():
    """Main function to run the analysis"""
    
    print("=== Pipeline Profile Analysis ===\n")
    
    # Create network
    print("Creating pipeline network...")
    network, nodes, pipes, positions = create_pipeline_profile_network()
    
    # Print network info
    print(f"\nNetwork created:")
    print(f"  Nodes: {len(nodes)}")
    print(f"  Pipes: {len(pipes)}")
    print(f"  Total length: {positions[-1]} m ({positions[-1]/1000:.1f} km)")
    
    # Solve network
    results, fluid = solve_network(network)
    
    if not results.converged:
        print("ERROR: Solution did not converge!")
        return
    
    print(f"\n✅ Solution converged in {results.iterations} iterations")
    
    # Create profile plots
    print("\nGenerating profile plots...")
    fig1 = plot_pipeline_profiles(network, nodes, pipes, positions, results, fluid)
    
    # Create 3D visualization
    print("Generating 3D visualization...")
    fig2 = plot_3d_profile(network, nodes, pipes, positions, results)
    
    # Save figures
    fig1.savefig('pipeline_profiles.png', dpi=300, bbox_inches='tight')
    fig2.savefig('pipeline_3d.png', dpi=300, bbox_inches='tight')
    print("\nPlots saved as 'pipeline_profiles.png' and 'pipeline_3d.png'")
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    main()