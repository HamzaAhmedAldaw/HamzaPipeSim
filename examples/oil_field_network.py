
// ===== examples/oil_field_network.py =====
# AI_GENERATED: Complex oil field network example
"""
Oil Field Network Simulation Example

This example demonstrates simulation of a realistic oil field gathering system
with multiple wells, manifolds, and processing facilities.
"""

import pipeline_sim as ps
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def create_oil_field_network():
    """Create a realistic oil field network"""
    network = ps.Network()
    
    # Wells (sources)
    wells = []
    well_data = [
        ("Well-A1", 0, 0, 150),      # (name, x_km, y_km, depth_m)
        ("Well-A2", 2, 0, 180),
        ("Well-A3", 4, 0, 200),
        ("Well-B1", 0, 3, 160),
        ("Well-B2", 2, 3, 170),
        ("Well-B3", 4, 3, 190),
        ("Well-C1", 1, 6, 140),
        ("Well-C2", 3, 6, 155),
    ]
    
    for name, x, y, depth in well_data:
        well = network.add_node(name, ps.NodeType.SOURCE)
        well.elevation = -depth  # Below surface
        wells.append((name, well, x, y))
    
    # Manifolds (junctions)
    manifold_a = network.add_node("Manifold-A", ps.NodeType.JUNCTION)
    manifold_a.elevation = 0
    
    manifold_b = network.add_node("Manifold-B", ps.NodeType.JUNCTION)
    manifold_b.elevation = 0
    
    manifold_c = network.add_node("Manifold-C", ps.NodeType.JUNCTION)
    manifold_c.elevation = 0
    
    central_manifold = network.add_node("Central-Manifold", ps.NodeType.JUNCTION)
    central_manifold.elevation = 5
    
    # Processing facilities
    separator = network.add_node("Separator", ps.NodeType.SEPARATOR)
    separator.elevation = 10
    
    storage = network.add_node("Storage", ps.NodeType.SINK)
    storage.elevation = 10
    
    # Connect wells to manifolds
    for name, well, x, y in wells:
        if "A" in name:
            target = manifold_a
        elif "B" in name:
            target = manifold_b
        else:
            target = manifold_c
            
        # Flowlines from wells
        pipe = network.add_pipe(
            f"FL-{name}",
            well,
            target,
            length=np.sqrt((x-2)**2 + (y-0)**2) * 1000,  # Convert km to m
            diameter=0.15
        )
        pipe.roughness = 0.000045
        pipe.inclination = np.arctan(depth / pipe.length)
    
    # Connect manifolds to central
    network.add_pipe("Trunk-A", manifold_a, central_manifold, 3000, 0.3)
    network.add_pipe("Trunk-B", manifold_b, central_manifold, 2500, 0.3)
    network.add_pipe("Trunk-C", manifold_c, central_manifold, 4000, 0.25)
    
    # Central to processing
    network.add_pipe("Main-Line", central_manifold, separator, 5000, 0.4)
    network.add_pipe("Sep-Storage", separator, storage, 500, 0.35)
    
    return network, wells


def set_well_conditions(network, wells):
    """Set production rates and pressures for wells"""
    
    # Well production data (pressure in bar, rate in m³/day)
    production_data = {
        "Well-A1": (180, 500),
        "Well-A2": (175, 450),
        "Well-A3": (170, 400),
        "Well-B1": (185, 550),
        "Well-B2": (182, 520),
        "Well-B3": (178, 480),
        "Well-C1": (190, 600),
        "Well-C2": (188, 580),
    }
    
    for name, (pressure_bar, rate_m3_day) in production_data.items():
        # Find well node
        well_node = network.get_node(name)
        if well_node:
            # Set bottomhole pressure
            network.set_pressure(well_node, pressure_bar * 1e5)
            
            # Convert daily rate to m³/s
            rate_m3_s = rate_m3_day / (24 * 3600)
            # Note: In practice, would set this as a well model
    
    # Set delivery pressure at storage
    storage = network.get_node("Storage")
    network.set_pressure(storage, 20e5)  # 20 bar


def create_fluid_model():
    """Create realistic fluid properties for oil field"""
    fluid = ps.FluidProperties()
    
    # Black oil properties
    fluid.oil_density = 820      # kg/m³ at standard conditions
    fluid.gas_density = 0.75     # Relative to air
    fluid.water_density = 1050   # Brine
    
    fluid.oil_viscosity = 0.008  # Pa.s (8 cP)
    fluid.gas_viscosity = 1.5e-5
    fluid.water_viscosity = 0.0012
    
    # Production characteristics
    fluid.gas_oil_ratio = 150    # sm³/sm³
    fluid.water_cut = 0.15       # 15% water
    fluid.api_gravity = 35       # Medium oil
    
    # Phase fractions (simplified)
    fluid.water_fraction = fluid.water_cut
    fluid.oil_fraction = (1 - fluid.water_cut) * 0.85
    fluid.gas_fraction = (1 - fluid.water_cut) * 0.15
    
    return fluid


def run_oil_field_simulation():
    """Run complete oil field simulation"""
    print("Oil Field Network Simulation")
    print("=" * 50)
    
    # Create network
    network, wells = create_oil_field_network()
    set_well_conditions(network, wells)
    fluid = create_fluid_model()
    
    print(f"Network created:")
    print(f"  Wells: {len(wells)}")
    print(f"  Total nodes: {len(network.nodes)}")
    print(f"  Total pipes: {len(network.pipes)}")
    
    # Run steady-state simulation
    print("\nRunning steady-state simulation...")
    solver = ps.SteadyStateSolver(network, fluid)
    solver.config.tolerance = 1e-6
    solver.config.max_iterations = 500
    solver.config.verbose = True
    
    results = solver.solve()
    
    if results.converged:
        print(f"\nSimulation converged in {results.iterations} iterations")
        print(f"Computation time: {results.computation_time:.2f} s")
        
        # Analyze results
        analyze_oil_field_results(network, results, fluid)
        
        # Generate visualizations
        visualize_oil_field(network, results, wells)
        
        # Save results
        ps.save_results(results, "oil_field_results.csv")
        ps.generate_report(network, results, fluid, "oil_field_report.html")
        
    else:
        print("Simulation failed to converge!")
        
    return network, results, fluid


def analyze_oil_field_results(network, results, fluid):
    """Analyze oil field simulation results"""
    
    print("\n" + "="*50)
    print("PRODUCTION ANALYSIS")
    print("="*50)
    
    # Well production rates
    print("\nWell Production Rates:")
    total_oil = 0
    total_gas = 0
    total_water = 0
    
    for well_name in ["Well-A1", "Well-A2", "Well-A3", "Well-B1", 
                      "Well-B2", "Well-B3", "Well-C1", "Well-C2"]:
        # Find flowline from well
        flowline_name = f"FL-{well_name}"
        if flowline_name in results.pipe_flow_rates:
            flow = results.pipe_flow_rates[flowline_name]
            
            # Convert to daily rates
            flow_m3_day = flow * 24 * 3600
            oil_rate = flow_m3_day * fluid.oil_fraction
            gas_rate = flow_m3_day * fluid.gas_fraction * fluid.gas_oil_ratio
            water_rate = flow_m3_day * fluid.water_fraction
            
            print(f"  {well_name}: {flow_m3_day:.1f} m³/day total")
            print(f"    Oil: {oil_rate:.1f} m³/day")
            print(f"    Gas: {gas_rate:.0f} sm³/day")
            print(f"    Water: {water_rate:.1f} m³/day")
            
            total_oil += oil_rate
            total_gas += gas_rate
            total_water += water_rate
    
    print(f"\nTotal Field Production:")
    print(f"  Oil: {total_oil:.1f} m³/day ({total_oil * 6.29:.0f} bbl/day)")
    print(f"  Gas: {total_gas:.0f} sm³/day ({total_gas * 35.31:.0f} scf/day)")
    print(f"  Water: {total_water:.1f} m³/day")
    
    # Pressure analysis
    print("\nPressure Analysis:")
    manifold_pressures = {}
    for manifold in ["Manifold-A", "Manifold-B", "Manifold-C", "Central-Manifold"]:
        if manifold in results.node_pressures:
            pressure = results.node_pressures[manifold] / 1e5  # Convert to bar
            manifold_pressures[manifold] = pressure
            print(f"  {manifold}: {pressure:.1f} bar")
    
    # Separator conditions
    if "Separator" in results.node_pressures:
        sep_pressure = results.node_pressures["Separator"] / 1e5
        print(f"  Separator: {sep_pressure:.1f} bar")
    
    # Flow assurance checks
    print("\nFlow Assurance Checks:")
    issues = []
    
    for pipe_id, pipe in network.pipes.items():
        flow = results.pipe_flow_rates.get(pipe_id, 0)
        velocity = flow / pipe.area()
        
        # Check velocity limits
        if velocity > 10:
            issues.append(f"High velocity in {pipe_id}: {velocity:.1f} m/s")
        elif velocity < 1:
            issues.append(f"Low velocity in {pipe_id}: {velocity:.1f} m/s (slug risk)")
        
        # Check pressure drop
        dp = results.pipe_pressure_drops.get(pipe_id, 0) / 1e5
        if dp > 10:
            issues.append(f"High pressure drop in {pipe_id}: {dp:.1f} bar")
    
    if issues:
        print("  Issues found:")
        for issue in issues[:5]:  # Show first 5 issues
            print(f"    - {issue}")
        if len(issues) > 5:
            print(f"    ... and {len(issues) - 5} more")
    else:
        print("  No flow assurance issues detected")
    
    # Line pack
    line_pack = ps.LinePackCalculator.calculate(network, results, fluid)
    print(f"\nLine Pack:")
    print(f"  Total mass: {line_pack.total_mass/1000:.1f} tonnes")
    print(f"  Total volume: {line_pack.total_volume:.1f} m³")
    print(f"  Average pressure: {line_pack.average_pressure/1e5:.1f} bar")


def visualize_oil_field(network, results, wells):
    """Create visualizations for oil field network"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Network layout
    ax1 = plt.subplot(2, 2, 1)
    
    # Plot network topology
    well_x = [w[2] for w in wells]
    well_y = [w[3] for w in wells]
    
    ax1.scatter(well_x, well_y, s=200, c='green', marker='^', label='Wells', zorder=3)
    
    # Add well labels
    for name, _, x, y in wells:
        ax1.annotate(name.split('-')[1], (x, y), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8)
    
    # Manifolds
    ax1.scatter([2], [0], s=300, c='blue', marker='s', label='Manifold A', zorder=3)
    ax1.scatter([2], [3], s=300, c='blue', marker='s', label='Manifold B', zorder=3)
    ax1.scatter([2], [6], s=300, c='blue', marker='s', label='Manifold C', zorder=3)
    ax1.scatter([2], [9], s=400, c='red', marker='D', label='Central', zorder=3)
    ax1.scatter([2], [12], s=400, c='orange', marker='o', label='Processing', zorder=3)
    
    # Draw pipes with flow indication
    for pipe_id, pipe in network.pipes.items():
        if "FL-" in pipe_id:  # Flowlines
            # Find well position
            well_name = pipe_id.replace("FL-", "")
            well_data = next((w for w in wells if w[0] == well_name), None)
            if well_data:
                x1, y1 = well_data[2], well_data[3]
                x2, y2 = 2, 0 if 'A' in well_name else (3 if 'B' in well_name else 6)
                ax1.plot([x1, x2], [y1, y2], 'k-', alpha=0.5, linewidth=1)
    
    # Main pipelines
    ax1.plot([2, 2], [0, 9], 'k-', linewidth=3, alpha=0.7)
    ax1.plot([2, 2], [3, 9], 'k-', linewidth=3, alpha=0.7)
    ax1.plot([2, 2], [6, 9], 'k-', linewidth=2.5, alpha=0.7)
    ax1.plot([2, 2], [9, 12], 'k-', linewidth=4, alpha=0.8)
    
    ax1.set_xlabel('Distance East (km)')
    ax1.set_ylabel('Distance North (km)')
    ax1.set_title('Oil Field Network Layout')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # 2. Pressure profile along main flow path
    ax2 = plt.subplot(2, 2, 2)
    
    # Extract pressures along path
    path_nodes = ["Well-B2", "Manifold-B", "Central-Manifold", "Separator", "Storage"]
    distances = [0, 2.5, 5.5, 10.5, 11.0]  # Approximate distances in km
    pressures = []
    
    for node in path_nodes:
        if node in results.node_pressures:
            pressures.append(results.node_pressures[node] / 1e5)
        else:
            pressures.append(np.nan)
    
    ax2.plot(distances, pressures, 'b-o', linewidth=2, markersize=8)
    ax2.set_xlabel('Distance along flow path (km)')
    ax2.set_ylabel('Pressure (bar)')
    ax2.set_title('Pressure Profile: Well-B2 to Storage')
    ax2.grid(True, alpha=0.3)
    
    # 3. Production by manifold
    ax3 = plt.subplot(2, 2, 3)
    
    manifold_flows = {"A": 0, "B": 0, "C": 0}
    for pipe_id, flow in results.pipe_flow_rates.items():
        if pipe_id.startswith("Trunk-"):
            manifold = pipe_id.split("-")[1]
            manifold_flows[manifold] = flow * 24 * 3600  # Convert to m³/day
    
    manifolds = list(manifold_flows.keys())
    flows = list(manifold_flows.values())
    
    bars = ax3.bar(manifolds, flows, color=['red', 'green', 'blue'])
    ax3.set_xlabel('Manifold')
    ax3.set_ylabel('Total Flow (m³/day)')
    ax3.set_title('Production by Manifold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, flow in zip(bars, flows):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(flow)}', ha='center', va='bottom')
    
    # 4. Velocity distribution
    ax4 = plt.subplot(2, 2, 4)
    
    velocities = []
    pipe_names = []
    
    for pipe_id, pipe in network.pipes.items():
        flow = results.pipe_flow_rates.get(pipe_id, 0)
        velocity = abs(flow) / pipe.area()
        velocities.append(velocity)
        pipe_names.append(pipe_id)
    
    # Sort by velocity
    sorted_data = sorted(zip(velocities, pipe_names), reverse=True)
    top_velocities = sorted_data[:10]  # Top 10
    
    v_values = [v[0] for v in top_velocities]
    v_names = [v[1] for v in top_velocities]
    
    bars = ax4.barh(range(len(v_values)), v_values)
    ax4.set_yticks(range(len(v_values)))
    ax4.set_yticklabels(v_names)
    ax4.set_xlabel('Velocity (m/s)')
    ax4.set_title('Top 10 Flow Velocities')
    ax4.axvline(x=10, color='r', linestyle='--', label='Max recommended')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Color bars by severity
    for bar, vel in zip(bars, v_values):
        if vel > 10:
            bar.set_color('red')
        elif vel > 7:
            bar.set_color('orange')
        else:
            bar.set_color('green')
    
    plt.tight_layout()
    plt.savefig('oil_field_analysis.png', dpi=150)
    plt.show()


def run_transient_analysis():
    """Run transient analysis for well shutdown scenario"""
    print("\n" + "="*50)
    print("TRANSIENT ANALYSIS - Well Shutdown")
    print("="*50)
    
    # Create network
    network, wells = create_oil_field_network()
    set_well_conditions(network, wells)
    fluid = create_fluid_model()
    
    # Create transient solver
    solver = ps.TransientSolver(network, fluid)
    solver.set_time_step(0.1)  # 0.1 second
    solver.set_simulation_time(300)  # 5 minutes
    solver.set_output_interval(1.0)  # Save every second
    
    # Add well shutdown event at t=30s
    # In practice, would implement proper well shutdown model
    # solver.add_event(ps.WellShutdownEvent("Well-B2", shutdown_time=30.0))
    
    print("Running transient simulation...")
    print("Scenario: Well-B2 emergency shutdown at t=30s")
    
    # results = solver.solve()
    
    # Plot transient results
    # plot_transient_results(solver.get_time_history())


if __name__ == "__main__":
    # Run main simulation
    network, results, fluid = run_oil_field_simulation()
    
    # Run transient analysis
    # run_transient_analysis()
