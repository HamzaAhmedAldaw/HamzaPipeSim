#!/usr/bin/env python3
"""
Pipeline Network Visualization
Creates visual representation of pipeline networks with flow results
"""

import sys
import os
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.patches import ArrowStyle
import matplotlib.cm as cm

# Load Pipeline-Sim
pyd_path = r"C:\Users\KIMO STORE\AppData\Roaming\Python\Python313\site-packages\pipeline_sim.cp313-win_amd64.pyd"
spec = importlib.util.spec_from_file_location("pipeline_sim", pyd_path)
pipeline_sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline_sim)

# Import classes
Network = pipeline_sim.Network
NodeType = pipeline_sim.NodeType
FluidProperties = pipeline_sim.FluidProperties
SteadyStateSolver = pipeline_sim.SteadyStateSolver

def create_complex_network():
    """Create a complex pipeline network"""
    
    network = Network()
    
    # Define network topology
    # Format: (name, x, y, type)
    node_data = [
        # Sources (wells)
        ("Well_A", 0, 3, NodeType.SOURCE),
        ("Well_B", 0, 1, NodeType.SOURCE),
        ("Well_C", 1, 4, NodeType.SOURCE),
        
        # Junctions
        ("Junc_1", 2, 3, NodeType.JUNCTION),
        ("Junc_2", 2, 1, NodeType.JUNCTION),
        ("Manifold", 4, 2, NodeType.JUNCTION),
        
        # Processing
        ("Separator", 6, 2, NodeType.JUNCTION),
        
        # Sinks
        ("Export", 8, 3, NodeType.SINK),
        ("Storage", 8, 1, NodeType.SINK),
    ]
    
    # Create nodes
    nodes = {}
    positions = {}
    for name, x, y, node_type in node_data:
        node = network.add_node(name, node_type)
        nodes[name] = node
        positions[name] = (x, y)
    
    # Define pipes
    # Format: (from, to, length_km, diameter_m)
    pipe_data = [
        ("Well_A", "Junc_1", 2.5, 0.3),
        ("Well_B", "Junc_2", 2.5, 0.25),
        ("Well_C", "Junc_1", 1.5, 0.3),
        ("Junc_1", "Manifold", 2.0, 0.4),
        ("Junc_2", "Manifold", 2.0, 0.35),
        ("Manifold", "Separator", 2.0, 0.5),
        ("Separator", "Export", 2.5, 0.45),
        ("Separator", "Storage", 2.5, 0.3),
    ]
    
    # Create pipes
    pipes = {}
    for from_node, to_node, length_km, diameter in pipe_data:
        pipe_name = f"{from_node}_to_{to_node}"
        pipe = network.add_pipe(pipe_name, nodes[from_node], nodes[to_node],
                               length_km * 1000, diameter)
        pipes[pipe_name] = pipe
    
    # Set boundary conditions
    # Well pressures
    network.set_pressure(nodes["Well_A"], 90e5)   # 90 bar
    network.set_pressure(nodes["Well_B"], 85e5)   # 85 bar
    network.set_pressure(nodes["Well_C"], 95e5)   # 95 bar
    
    # Delivery pressures
    network.set_pressure(nodes["Export"], 50e5)   # 50 bar
    network.set_pressure(nodes["Storage"], 30e5)  # 30 bar
    
    return network, nodes, pipes, positions, node_data, pipe_data

def solve_network(network):
    """Solve the network"""
    
    # Fluid properties (light crude)
    fluid = FluidProperties()
    fluid.oil_fraction = 0.9
    fluid.water_fraction = 0.1
    fluid.oil_density = 820.0
    fluid.water_density = 1020.0
    fluid.oil_viscosity = 0.003
    fluid.water_viscosity = 0.001
    
    # Solver
    solver = SteadyStateSolver(network, fluid)
    config = solver.config
    config.verbose = True
    solver.set_config(config)
    
    print("\nSolving network...")
    results = solver.solve()
    
    return results, fluid

def visualize_network(network, nodes, pipes, positions, node_data, pipe_data, results):
    """Create network visualization"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Common visualization function
    def draw_network(ax, color_by='pressure'):
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Draw pipes
        for from_node, to_node, length_km, diameter in pipe_data:
            pipe_name = f"{from_node}_to_{to_node}"
            
            # Get positions
            x1, y1 = positions[from_node]
            x2, y2 = positions[to_node]
            
            # Get flow data
            flow = results.pipe_flow_rates[pipe_name]
            velocity = results.pipe_velocities[pipe_name]
            
            # Determine color based on parameter
            if color_by == 'pressure':
                # Use average pressure
                p1 = results.node_pressures[from_node] / 1e5
                p2 = results.node_pressures[to_node] / 1e5
                value = (p1 + p2) / 2
                cmap = plt.cm.coolwarm
                norm = plt.Normalize(30, 100)
            else:  # velocity
                value = velocity
                cmap = plt.cm.viridis
                norm = plt.Normalize(0, 15)
            
            color = cmap(norm(value))
            
            # Draw pipe with thickness proportional to diameter
            line_width = diameter * 20
            
            # Arrow for flow direction
            if flow > 0:
                arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                      connectionstyle="arc3,rad=0",
                                      arrowstyle=ArrowStyle('->', head_width=0.3, head_length=0.3),
                                      color=color, linewidth=line_width,
                                      alpha=0.8, zorder=1)
            else:
                arrow = FancyArrowPatch((x2, y2), (x1, y1),
                                      connectionstyle="arc3,rad=0",
                                      arrowstyle=ArrowStyle('->', head_width=0.3, head_length=0.3),
                                      color=color, linewidth=line_width,
                                      alpha=0.8, zorder=1)
            ax.add_patch(arrow)
            
            # Add flow label
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y, f'{abs(flow)*3600:.0f} m³/h',
                   fontsize=9, ha='center', va='bottom',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        # Draw nodes
        for name, x, y, node_type in node_data:
            pressure = results.node_pressures[name] / 1e5
            
            # Node appearance based on type
            if node_type == NodeType.SOURCE:
                marker = 's'
                size = 800
                color = 'green'
                edge = 'darkgreen'
            elif node_type == NodeType.SINK:
                marker = 'v'
                size = 800
                color = 'red'
                edge = 'darkred'
            else:  # JUNCTION
                marker = 'o'
                size = 600
                color = 'lightblue'
                edge = 'blue'
            
            ax.scatter(x, y, s=size, c=color, marker=marker,
                      edgecolors=edge, linewidth=2, zorder=3)
            
            # Node labels
            ax.text(x, y-0.3, f'{name}\n{pressure:.1f} bar',
                   fontsize=10, ha='center', va='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # Colorbar
        if color_by == 'pressure':
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Pressure (bar)', fontsize=12)
        else:
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Velocity (m/s)', fontsize=12)
    
    # Left plot - colored by pressure
    ax1.set_title('Network Colored by Pressure', fontsize=14, fontweight='bold')
    draw_network(ax1, 'pressure')
    ax1.set_xlabel('Distance (km)', fontsize=12)
    ax1.set_ylabel('Distance (km)', fontsize=12)
    
    # Right plot - colored by velocity
    ax2.set_title('Network Colored by Velocity', fontsize=14, fontweight='bold')
    draw_network(ax2, 'velocity')
    ax2.set_xlabel('Distance (km)', fontsize=12)
    ax2.set_ylabel('Distance (km)', fontsize=12)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='green', 
               markersize=12, label='Source (Well)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
               markersize=12, label='Junction'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='red', 
               markersize=12, label='Sink (Export/Storage)'),
    ]
    ax1.legend(handles=legend_elements, loc='upper left')
    
    plt.suptitle('Pipeline Network Flow Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def create_flow_balance_chart(results, node_data):
    """Create flow balance chart for each node"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    node_names = [name for name, _, _, _ in node_data]
    inflows = []
    outflows = []
    
    # Calculate flows for each node
    for node_name in node_names:
        inflow = 0
        outflow = 0
        
        # Check all pipes
        for pipe_name, flow in results.pipe_flow_rates.items():
            if f"to_{node_name}" in pipe_name:
                inflow += abs(flow) * 3600  # Convert to m³/h
            elif f"{node_name}_to" in pipe_name:
                outflow += abs(flow) * 3600
        
        inflows.append(inflow)
        outflows.append(outflow)
    
    # Create grouped bar chart
    x = np.arange(len(node_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, inflows, width, label='Inflow', alpha=0.8, color='green')
    bars2 = ax.bar(x + width/2, outflows, width, label='Outflow', alpha=0.8, color='red')
    
    ax.set_xlabel('Nodes', fontsize=12)
    ax.set_ylabel('Flow Rate (m³/h)', fontsize=12)
    ax.set_title('Flow Balance at Each Node', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(node_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add balance values
    for i, (inf, outf) in enumerate(zip(inflows, outflows)):
        balance = inf - outf
        if abs(balance) > 0.1:
            ax.text(i, max(inf, outf) + 50, f'Δ={balance:.1f}',
                   ha='center', va='bottom', fontsize=10,
                   color='red' if abs(balance) > 10 else 'green')
    
    plt.tight_layout()
    return fig

def main():
    """Run network visualization"""
    
    print("=== Pipeline Network Visualization ===")
    
    # Create network
    network, nodes, pipes, positions, node_data, pipe_data = create_complex_network()
    
    print(f"\nNetwork created:")
    print(f"  Nodes: {len(nodes)}")
    print(f"  Pipes: {len(pipes)}")
    print(f"  Sources: {sum(1 for _, _, _, t in node_data if t == NodeType.SOURCE)}")
    print(f"  Sinks: {sum(1 for _, _, _, t in node_data if t == NodeType.SINK)}")
    
    # Solve
    results, fluid = solve_network(network)
    
    if not results.converged:
        print("ERROR: Solution did not converge!")
        return
    
    print(f"\n✅ Solution converged in {results.iterations} iterations")
    
    # Calculate total production
    total_production = 0
    for name, _, _, node_type in node_data:
        if node_type == NodeType.SOURCE:
            # Sum outgoing flows from sources
            for pipe_name, flow in results.pipe_flow_rates.items():
                if pipe_name.startswith(name):
                    total_production += flow
    
    print(f"\nTotal production: {total_production:.3f} m³/s ({total_production*86400:.0f} m³/day)")
    
    # Create visualizations
    print("\nGenerating network visualization...")
    fig1 = visualize_network(network, nodes, pipes, positions, node_data, pipe_data, results)
    
    print("Generating flow balance chart...")
    fig2 = create_flow_balance_chart(results, node_data)
    
    # Save
    fig1.savefig('network_visualization.png', dpi=300, bbox_inches='tight')
    fig2.savefig('flow_balance.png', dpi=300, bbox_inches='tight')
    print("\nPlots saved as 'network_visualization.png' and 'flow_balance.png'")
    
    plt.show()

if __name__ == "__main__":
    main()