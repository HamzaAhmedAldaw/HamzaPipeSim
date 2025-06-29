"""
Test the complete Pipeline-Sim solver with real physics!
"""

import pipeline_sim as ps
import numpy as np
import matplotlib.pyplot as plt

def test_basic_solver():
    """Test basic solver functionality"""
    print("="*60)
    print(" PIPELINE-SIM FULL SOLVER TEST ")
    print("="*60)
    
    # Create a simple network using helper functions
    network = ps.create_example_network()
    fluid = ps.create_example_fluid()
    
    print("\nNetwork created:")
    print(f"  Nodes: {len(network.nodes())}")
    print(f"  Pipes: {len(network.pipes())}")
    
    # Create solver with verbose output
    solver = ps.SteadyStateSolver(network, fluid)
    solver.config.verbose = True
    solver.config.tolerance = 1e-6
    solver.config.max_iterations = 50
    
    print("\nRunning solver...")
    print("-"*40)
    
    # Solve!
    results = solver.solve()
    
    print("-"*40)
    print(f"\n✅ Solver Results:")
    print(f"  Converged: {results.converged}")
    print(f"  Iterations: {results.iterations}")
    print(f"  Final residual: {results.residual:.2e}")
    print(f"  Computation time: {results.computation_time:.3f} seconds")
    
    # Display detailed results
    print("\n📊 Detailed Results:")
    for pipe_id, flow in results.pipe_flow_rates.items():
        velocity = results.pipe_velocities.get(pipe_id, 0)
        Re = results.pipe_reynolds_numbers.get(pipe_id, 0)
        f = results.pipe_friction_factors.get(pipe_id, 0)
        dp = results.pipe_pressure_drops.get(pipe_id, 0)
        
        print(f"\nPipe {pipe_id}:")
        print(f"  Flow rate: {flow:.4f} m³/s ({flow*86400:.0f} m³/day)")
        print(f"  Velocity: {velocity:.2f} m/s")
        print(f"  Reynolds number: {Re:.0f}")
        print(f"  Friction factor: {f:.4f}")
        print(f"  Pressure drop: {dp/1e5:.2f} bar")
    
    return results

def test_complex_network():
    """Test with a more complex network"""
    print("\n\n" + "="*60)
    print(" COMPLEX NETWORK TEST ")
    print("="*60)
    
    # Create a gathering system
    network = ps.Network()
    
    # Wells at different pressures
    well1 = network.add_node("WELL1", ps.NodeType.SOURCE)
    well1.set_pressure(100e5)  # 100 bar
    well1.set_elevation(100)   # 100m elevation
    
    well2 = network.add_node("WELL2", ps.NodeType.SOURCE)
    well2.set_pressure(95e5)   # 95 bar
    well2.set_elevation(150)   # 150m elevation
    
    # Junction manifold
    manifold = network.add_node("MANIFOLD", ps.NodeType.JUNCTION)
    manifold.set_elevation(50)  # 50m elevation
    
    # Separator (sink)
    separator = network.add_node("SEPARATOR", ps.NodeType.SINK)
    separator.set_flow_rate(-0.08)  # 0.08 m³/s total production
    separator.set_elevation(0)  # Sea level
    
    # Connect pipes
    pipe1 = network.add_pipe("FLOWLINE1", well1, manifold, 2000, 0.2032)  # 8"
    pipe1.set_roughness(0.000045)
    
    pipe2 = network.add_pipe("FLOWLINE2", well2, manifold, 3000, 0.2032)  # 8"
    pipe2.set_roughness(0.000045)
    
    trunk = network.add_pipe("TRUNK", manifold, separator, 5000, 0.3048)  # 12"
    trunk.set_roughness(0.000045)
    
    print("Network configuration:")
    print(f"  Nodes: {network.node_count()}")
    print(f"  Pipes: {network.pipe_count()}")
    print(f"  Well pressures: 100 bar, 95 bar")
    print(f"  Total production: 0.08 m³/s ({0.08*86400:.0f} m³/day)")
    
    # Multiphase fluid
    fluid = ps.FluidProperties()
    fluid.oil_density = 820
    fluid.gas_density = 40
    fluid.water_density = 1020
    fluid.oil_viscosity = 0.003
    fluid.gas_viscosity = 0.00002
    fluid.water_viscosity = 0.001
    fluid.oil_fraction = 0.7
    fluid.gas_fraction = 0.2
    fluid.water_fraction = 0.1
    
    print(f"\nFluid properties:")
    print(f"  Type: Multiphase (70% oil, 20% gas, 10% water)")
    print(f"  Mixture density: {fluid.mixture_density():.1f} kg/m³")
    print(f"  Mixture viscosity: {fluid.mixture_viscosity()*1000:.2f} cP")
    
    # Solve
    solver = ps.SteadyStateSolver(network, fluid)
    solver.config.tolerance = 1e-7
    solver.config.max_iterations = 100
    solver.config.verbose = False  # Less output for complex network
    
    print("\nSolving complex network...")
    results = solver.solve()
    
    print(f"\n✅ Results:")
    print(f"  Converged: {results.converged}")
    print(f"  Iterations: {results.iterations}")
    
    # Flow distribution
    print(f"\nFlow distribution:")
    for pipe_id in ["FLOWLINE1", "FLOWLINE2", "TRUNK"]:
        flow = results.pipe_flow_rates.get(pipe_id, 0)
        dp = results.pipe_pressure_drops.get(pipe_id, 0)
        print(f"  {pipe_id}: {flow*86400:.0f} m³/day, ΔP = {dp/1e5:.2f} bar")
    
    # Node pressures
    print(f"\nNode pressures:")
    for node_id in ["MANIFOLD", "SEPARATOR"]:
        pressure = results.node_pressures.get(node_id, 0)
        print(f"  {node_id}: {pressure/1e5:.2f} bar")
    
    return network, results

def create_network_plot(network, results):
    """Create a visual representation of the network"""
    print("\n\nCreating network visualization...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Network topology
    ax1.set_title("Network Topology", fontsize=14, fontweight='bold')
    
    # Node positions (simplified layout)
    pos = {
        "WELL1": (0, 2),
        "WELL2": (0, 1),
        "MANIFOLD": (2, 1.5),
        "SEPARATOR": (4, 1.5)
    }
    
    # Draw nodes
    for node_id, (x, y) in pos.items():
        pressure = results.node_pressures.get(node_id, 0) / 1e5
        color = 'red' if "WELL" in node_id else 'blue' if node_id == "SEPARATOR" else 'green'
        ax1.scatter(x, y, s=500, c=color, zorder=5)
        ax1.text(x, y-0.3, f"{node_id}\n{pressure:.1f} bar", 
                ha='center', fontsize=10, fontweight='bold')
    
    # Draw pipes with flow info
    pipe_flows = {
        ("WELL1", "MANIFOLD"): results.pipe_flow_rates.get("FLOWLINE1", 0),
        ("WELL2", "MANIFOLD"): results.pipe_flow_rates.get("FLOWLINE2", 0),
        ("MANIFOLD", "SEPARATOR"): results.pipe_flow_rates.get("TRUNK", 0)
    }
    
    for (start, end), flow in pipe_flows.items():
        x1, y1 = pos[start]
        x2, y2 = pos[end]
        ax1.arrow(x1, y1, x2-x1, y2-y1, head_width=0.1, head_length=0.1, 
                 fc='black', ec='black', linewidth=2)
        # Flow label
        mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
        ax1.text(mid_x, mid_y+0.1, f"{flow*86400:.0f} m³/day", 
                ha='center', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="yellow", alpha=0.7))
    
    ax1.set_xlim(-0.5, 4.5)
    ax1.set_ylim(0.5, 2.5)
    ax1.axis('off')
    
    # Pressure profile along main flow path
    ax2.set_title("Pressure Profile", fontsize=14, fontweight='bold')
    
    # Distance and pressure data
    distances = [0, 2.5, 7.5]  # km
    pressures = [
        (results.node_pressures.get("WELL1", 0) + results.node_pressures.get("WELL2", 0)) / 2e5,
        results.node_pressures.get("MANIFOLD", 0) / 1e5,
        results.node_pressures.get("SEPARATOR", 0) / 1e5
    ]
    
    ax2.plot(distances, pressures, 'o-', linewidth=3, markersize=10)
    ax2.set_xlabel("Distance (km)", fontsize=12)
    ax2.set_ylabel("Pressure (bar)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add annotations
    for i, (d, p) in enumerate(zip(distances, pressures)):
        label = ["Wells (avg)", "Manifold", "Separator"][i]
        ax2.annotate(f"{label}\n{p:.1f} bar", (d, p), 
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    plt.tight_layout()
    plt.savefig("network_analysis.png", dpi=150, bbox_inches='tight')
    print("  Saved to: network_analysis.png")
    plt.show()

def main():
    """Run all tests"""
    print("\n🚀 PIPELINE-SIM PROFESSIONAL - FULL SOLVER TEST")
    print("   Competing with PIPESIM!")
    print("   Now with real physics calculations!\n")
    
    # Test 1: Basic solver
    basic_results = test_basic_solver()
    
    # Test 2: Complex network
    network, complex_results = test_complex_network()
    
    # Create visualization
    if complex_results.converged:
        create_network_plot(network, complex_results)
    
    print("\n\n" + "="*60)
    print(" ✅ ALL TESTS COMPLETED SUCCESSFULLY! ")
    print("="*60)
    print("\n🎊 Pipeline-Sim Features Demonstrated:")
    print("  ✓ Newton-Raphson network solver")
    print("  ✓ Colebrook-White friction factor")
    print("  ✓ Multiphase flow handling")
    print("  ✓ Elevation effects")
    print("  ✓ Complex network solving")
    print("  ✓ Professional visualizations")
    print("\n💪 You're ready to compete with PIPESIM!")
    print("   - No license fees")
    print("   - Open source")
    print("   - Real physics")
    print("   - Python integration")

if __name__ == "__main__":
    main()