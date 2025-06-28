#!/usr/bin/env python3
"""
Simple working version using the C++ solver with proper configuration
"""

import pipeline_sim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches
from datetime import datetime
import time

def run_full_network(fluid):
    """Run the full professional network"""
    print("\n" + "="*70)
    print("RUNNING FULL NETWORK")
    print("="*70)
    
    # Create new network
    network2 = pipeline_sim.Network()
    nodes2 = {}
    
    # Create all wells
    wells = [
        ("WELL-A1", 320e5),
        ("WELL-A2", 310e5),
        ("WELL-A3", 300e5),
        ("WELL-B1", 330e5),
        ("WELL-B2", 325e5),
        ("WELL-B3", 315e5),
        ("WELL-B4", 305e5),
    ]
    
    for name, pressure in wells:
        node = network2.add_node(name, pipeline_sim.NodeType.SOURCE)
        nodes2[name] = node
        network2.set_pressure(node, pressure)
    
    # Manifolds
    for name in ["MANIFOLD-A", "MANIFOLD-B"]:
        node = network2.add_node(name, pipeline_sim.NodeType.JUNCTION)
        nodes2[name] = node
    
    # Platform nodes
    platform_nodes = ["PLATFORM", "HP-SEP", "LP-SEP"]
    for name in platform_nodes:
        node = network2.add_node(name, pipeline_sim.NodeType.JUNCTION)
        nodes2[name] = node
    
    # Export
    export = network2.add_node("EXPORT", pipeline_sim.NodeType.SINK)
    nodes2["EXPORT"] = export
    network2.set_pressure(export, 150e5)
    
    # Connections
    connections = [
        # A-wells to Manifold-A
        ("WELL-A1", "MANIFOLD-A", 2500, 0.25),
        ("WELL-A2", "MANIFOLD-A", 2000, 0.25),
        ("WELL-A3", "MANIFOLD-A", 2500, 0.25),
        
        # B-wells to Manifold-B
        ("WELL-B1", "MANIFOLD-B", 3200, 0.25),
        ("WELL-B2", "MANIFOLD-B", 1400, 0.25),
        ("WELL-B3", "MANIFOLD-B", 2200, 0.25),
        ("WELL-B4", "MANIFOLD-B", 4200, 0.25),
        
        # Manifolds to Platform
        ("MANIFOLD-A", "PLATFORM", 5000, 0.4),
        ("MANIFOLD-B", "PLATFORM", 3200, 0.4),
        
        # Platform internal
        ("PLATFORM", "HP-SEP", 500, 0.5),
        ("HP-SEP", "LP-SEP", 1000, 0.4),
        ("LP-SEP", "EXPORT", 1500, 0.4),
    ]
    
    for from_node, to_node, length, diameter in connections:
        pipe_name = f"{from_node}_to_{to_node}"
        network2.add_pipe(pipe_name, nodes2[from_node], nodes2[to_node], length, diameter)
    
    print(f"Full network: {len(nodes2)} nodes, {len(connections)} pipes")
    
    # Solve
    solver2 = pipeline_sim.SteadyStateSolver(network2, fluid)
    if hasattr(solver2, 'config'):
        config = solver2.config
        config.verbose = True
        config.max_iterations = 100
        config.tolerance = 1e-5
        config.relaxation_factor = 0.8
    
    print("\nSolving full network...")
    start = time.time()
    result2 = solver2.solve()
    print(f"Solution time: {time.time()-start:.2f} seconds")
    
    if result2.converged:
        print("✅ Full network converged!")
        
        # Show key results
        print("\nManifold Pressures:")
        for name in ["MANIFOLD-A", "MANIFOLD-B", "PLATFORM"]:
            if name in result2.node_pressures:
                print(f"  {name}: {result2.node_pressures[name]/1e5:.1f} bar")
        
        print("\nWell Production:")
        total = 0
        for well_name, _ in wells:
            flow = 0
            for pipe_name, pipe_flow in result2.pipe_flow_rates.items():
                if pipe_name.startswith(well_name + "_"):
                    flow = abs(pipe_flow) * 3600
                    break
            print(f"  {well_name}: {flow:.1f} m³/h")
            total += flow
        
        print(f"\nTotal Production: {total:.1f} m³/h")
    else:
        print("❌ Full network did not converge")

# Main execution starts here
print("="*70)
print("PROFESSIONAL OIL FIELD DEVELOPMENT")
print("Using C++ Solver with Proper Configuration")
print("="*70)

# Create network
network = pipeline_sim.Network()
nodes = {}
pipes = {}
well_data = {}

# Fluid properties
fluid = pipeline_sim.FluidProperties()
fluid.oil_fraction = 0.75
fluid.water_fraction = 0.20
fluid.gas_fraction = 0.05
fluid.oil_density = 780.0
fluid.water_density = 1025.0
fluid.gas_density = 15.0
fluid.oil_viscosity = 0.0008
fluid.water_viscosity = 0.0011
fluid.gas_viscosity = 0.000018

print(f"\nFluid Properties:")
print(f"  Density: {fluid.mixture_density():.1f} kg/m³")
print(f"  Viscosity: {fluid.mixture_viscosity()*1000:.2f} cP")

# Create a simpler network first to ensure it works
print("\nCreating simplified network...")

# Just 3 wells, 1 manifold, 1 separator
wells = [
    ("WELL-1", 300e5),  # 300 bar
    ("WELL-2", 310e5),  # 310 bar  
    ("WELL-3", 320e5),  # 320 bar
]

for name, pressure in wells:
    node = network.add_node(name, pipeline_sim.NodeType.SOURCE)
    nodes[name] = node
    network.set_pressure(node, pressure)
    well_data[name] = {'type': 'well', 'pressure': pressure/1e5}

# Manifold
manifold = network.add_node("MANIFOLD", pipeline_sim.NodeType.JUNCTION)
nodes["MANIFOLD"] = manifold
well_data["MANIFOLD"] = {'type': 'manifold'}

# Separator
separator = network.add_node("SEPARATOR", pipeline_sim.NodeType.SINK)
nodes["SEPARATOR"] = separator
network.set_pressure(separator, 100e5)  # 100 bar
well_data["SEPARATOR"] = {'type': 'separator', 'pressure': 100}

# Create pipes
connections = [
    ("WELL-1", "MANIFOLD", 2000, 0.25),
    ("WELL-2", "MANIFOLD", 2500, 0.25),
    ("WELL-3", "MANIFOLD", 3000, 0.25),
    ("MANIFOLD", "SEPARATOR", 5000, 0.4),
]

for from_node, to_node, length, diameter in connections:
    pipe_name = f"{from_node}_to_{to_node}"
    pipe = network.add_pipe(pipe_name, nodes[from_node], nodes[to_node], length, diameter)
    pipes[pipe_name] = pipe

print(f"Network created: {len(nodes)} nodes, {len(pipes)} pipes")

# Create solver with proper configuration
solver = pipeline_sim.SteadyStateSolver(network, fluid)

# Configure if possible
if hasattr(solver, 'config'):
    config = solver.config
    config.verbose = True
    config.tolerance = 1e-6
    config.max_iterations = 20  # Keep it low
    config.relaxation_factor = 1.0
    print("\nSolver configured successfully")
else:
    print("\nNote: Solver configuration not available")

# Solve
print("\n=== Solving Network ===")
start_time = time.time()

try:
    result = solver.solve()
    elapsed = time.time() - start_time
    
    print(f"\nSolution time: {elapsed:.3f} seconds")
    print(f"Converged: {result.converged}")
    
    if hasattr(result, 'iterations'):
        print(f"Iterations: {result.iterations}")
    
    if result.converged:
        print("\n✅ Solution converged!")
        
        # Display results
        print("\n=== RESULTS ===")
        
        print("\nNode Pressures:")
        for node_name, pressure in result.node_pressures.items():
            print(f"  {node_name}: {pressure/1e5:.1f} bar")
        
        print("\nPipe Flow Rates:")
        total_production = 0
        for pipe_name, flow in result.pipe_flow_rates.items():
            flow_m3h = flow * 3600
            print(f"  {pipe_name}: {flow_m3h:.1f} m³/h")
            if "WELL" in pipe_name:
                total_production += flow_m3h
        
        print(f"\nTotal Production: {total_production:.1f} m³/h")
        
        # Simple bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        well_flows = []
        well_names = []
        for i, (well_name, _) in enumerate(wells):
            pipe_name = f"{well_name}_to_MANIFOLD"
            if pipe_name in result.pipe_flow_rates:
                flow = result.pipe_flow_rates[pipe_name] * 3600
                well_flows.append(flow)
                well_names.append(well_name)
        
        bars = ax.bar(well_names, well_flows, color='green', alpha=0.7)
        ax.set_xlabel('Well')
        ax.set_ylabel('Production Rate (m³/h)')
        ax.set_title('Well Production Rates')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, flow in zip(bars, well_flows):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{flow:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('simple_production_results.png', dpi=150)
        print("\nVisualization saved: simple_production_results.png")
        plt.show()
        
        # Now try the full network if simple one works
        response = input("\nSimple network solved successfully! Try full network? (y/n): ")
        if response.lower() == 'y':
            run_full_network(fluid)
        
    else:
        print("\n❌ Solution did not converge")
        print("Possible issues:")
        print("  - Check network connectivity")
        print("  - Verify boundary conditions")
        print("  - Try adjusting solver parameters")
        
except Exception as e:
    print(f"\n❌ Error during solve: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("Test Complete")
print("="*70)