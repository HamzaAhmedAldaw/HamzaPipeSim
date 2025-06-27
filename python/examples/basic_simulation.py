# ===== python/examples/basic_simulation.py =====
# AI_GENERATED: Basic pipeline simulation example
import pipeline_sim as ps
import matplotlib.pyplot as plt
import numpy as np


def main():
    """Run basic pipeline simulation"""
    
    print("Pipeline-Sim Basic Example")
    print("=" * 40)
    
    # Create network
    network = ps.Network()
    
    # Define nodes
    wellhead = network.add_node("wellhead", ps.NodeType.SOURCE)
    manifold = network.add_node("manifold", ps.NodeType.JUNCTION)
    separator = network.add_node("separator", ps.NodeType.SINK)
    
    # Set elevations
    wellhead.elevation = 0
    manifold.elevation = 50
    separator.elevation = 30
    
    # Define pipes
    riser = network.add_pipe("riser", wellhead, manifold, 
                            length=1500, diameter=0.3)
    flowline = network.add_pipe("flowline", manifold, separator,
                               length=5000, diameter=0.4)
    
    # Set pipe properties
    riser.roughness = 0.000045  # Commercial steel
    riser.inclination = np.pi/4  # 45 degrees
    
    flowline.roughness = 0.000045
    flowline.inclination = -0.02  # Slight downward slope
    
    # Define fluid properties
    fluid = ps.FluidProperties()
    fluid.oil_density = 850  # kg/m³
    fluid.gas_density = 0.8  # relative
    fluid.water_density = 1025
    fluid.oil_viscosity = 0.005  # Pa.s
    fluid.gas_viscosity = 1.5e-5
    fluid.water_viscosity = 0.001
    
    # Multiphase properties
    fluid.gas_oil_ratio = 150  # sm³/sm³
    fluid.water_cut = 0.3  # 30% water
    
    # Calculate phase fractions
    fluid.water_fraction = fluid.water_cut
    fluid.oil_fraction = (1 - fluid.water_cut) * 0.9  # Approximate
    fluid.gas_fraction = 0.1  # Approximate
    
    print(f"\nFluid Properties:")
    print(f"  Mixture density: {fluid.mixture_density():.1f} kg/m³")
    print(f"  Mixture viscosity: {fluid.mixture_viscosity()*1000:.2f} cP")
    
    # Set boundary conditions
    network.set_pressure(wellhead, 70e5)  # 70 bar
    network.set_flow_rate(separator, 0.1)  # 0.1 m³/s
    
    # Create solver
    solver = ps.SteadyStateSolver(network, fluid)
    solver.config.tolerance = 1e-6
    solver.config.max_iterations = 100
    solver.config.verbose = True
    
    print("\nRunning simulation...")
    results = solver.solve()
    
    # Display results
    print(f"\nSimulation {'converged' if results.converged else 'failed'}!")
    print(f"Iterations: {results.iterations}")
    print(f"Residual: {results.residual:.2e}")
    
    print("\nNode Pressures:")
    for node_id, pressure in results.node_pressures.items():
        print(f"  {node_id}: {pressure/1e5:.2f} bar")
    
    print("\nPipe Flow Rates:")
    for pipe_id, flow in results.pipe_flow_rates.items():
        velocity = flow / network.pipes[pipe_id].area()
        print(f"  {pipe_id}: {flow:.4f} m³/s ({velocity:.2f} m/s)")
    
    print("\nPressure Drops:")
    for pipe_id, dp in results.pipe_pressure_drops.items():
        print(f"  {pipe_id}: {dp/1e5:.3f} bar")
    
    # Generate plots
    plot_pressure_profile(network, results)
    
    # Save results
    ps.save_results(results, "simulation_results.csv")
    ps.generate_report(network, results, fluid, "simulation_report.html")
    
    print("\nResults saved to simulation_results.csv and simulation_report.html")


def plot_pressure_profile(network, results):
    """Plot pressure profile along pipeline"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Pressure profile
    nodes = ['wellhead', 'manifold', 'separator']
    positions = [0, 1500, 6500]  # Cumulative distances
    pressures = [results.node_pressures[n]/1e5 for n in nodes]
    
    ax1.plot(positions, pressures, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Pressure (bar)')
    ax1.set_title('Pressure Profile')
    ax1.grid(True, alpha=0.3)
    
    # Elevation profile
    elevations = [network.nodes[n].elevation for n in nodes]
    ax2.plot(positions, elevations, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Elevation (m)')
    ax2.set_title('Elevation Profile')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pressure_profile.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
