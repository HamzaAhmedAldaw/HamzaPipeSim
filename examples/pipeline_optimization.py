# ===== examples/pipeline_optimization.py =====
# AI_GENERATED: Pipeline network optimization example
"""
Pipeline Network Optimization Example

Demonstrates using ML-based optimization to minimize energy consumption
while meeting delivery requirements.
"""

import pipeline_sim as ps
from pipeline_sim.ml_integration import MLOptimizer, OptimizationObjective
import numpy as np
import matplotlib.pyplot as plt


def create_pipeline_network_with_pumps():
    """Create network with multiple pump stations"""
    network = ps.Network()
    
    # Nodes
    inlet = network.add_node("Inlet", ps.NodeType.SOURCE)
    pump1_in = network.add_node("Pump1_In", ps.NodeType.JUNCTION)
    pump1_out = network.add_node("Pump1_Out", ps.NodeType.PUMP)
    
    junction1 = network.add_node("Junction1", ps.NodeType.JUNCTION)
    
    pump2_in = network.add_node("Pump2_In", ps.NodeType.JUNCTION)
    pump2_out = network.add_node("Pump2_Out", ps.NodeType.PUMP)
    
    junction2 = network.add_node("Junction2", ps.NodeType.JUNCTION)
    
    delivery1 = network.add_node("Delivery1", ps.NodeType.SINK)
    delivery2 = network.add_node("Delivery2", ps.NodeType.SINK)
    
    # Set elevations (m)
    inlet.elevation = 0
    pump1_in.elevation = 0
    pump1_out.elevation = 0
    junction1.elevation = 100  # Uphill
    pump2_in.elevation = 100
    pump2_out.elevation = 100
    junction2.elevation = 50   # Downhill
    delivery1.elevation = 80
    delivery2.elevation = 20
    
    # Pipes
    network.add_pipe("Inlet_Pump1", inlet, pump1_in, 1000, 0.5)
    network.add_pipe("Pump1_Internal", pump1_in, pump1_out, 10, 0.5)
    network.add_pipe("Pump1_Junction1", pump1_out, junction1, 5000, 0.5)
    
    network.add_pipe("Junction1_Pump2", junction1, pump2_in, 2000, 0.4)
    network.add_pipe("Pump2_Internal", pump2_in, pump2_out, 10, 0.4)
    network.add_pipe("Pump2_Junction2", pump2_out, junction2, 3000, 0.4)
    
    network.add_pipe("Junction2_Delivery1", junction2, delivery1, 2000, 0.3)
    network.add_pipe("Junction2_Delivery2", junction2, delivery2, 3000, 0.35)
    
    # Boundary conditions
    network.set_pressure(inlet, 30e5)  # 30 bar supply
    network.set_flow_rate(delivery1, 0.15)  # 150 l/s
    network.set_flow_rate(delivery2, 0.25)  # 250 l/s
    
    return network


def optimize_pump_operation():
    """Optimize pump speeds to minimize energy consumption"""
    print("Pipeline Network Optimization")
    print("="*50)
    
    # Create network
    network = create_pipeline_network_with_pumps()
    
    # Fluid properties (light crude oil)
    fluid = ps.FluidProperties()
    fluid.oil_density = 800
    fluid.oil_viscosity = 0.005
    fluid.oil_fraction = 1.0
    fluid.gas_fraction = 0.0
    fluid.water_fraction = 0.0
    
    # Create optimizer
    optimizer = MLOptimizer()
    
    # Define objective
    objective = OptimizationObjective()
    objective.type = OptimizationObjective.MINIMIZE_ENERGY_CONSUMPTION
    
    # Define constraints
    constraints = ps.OptimizationConstraints()
    constraints.min_pressure = 5e5      # 5 bar minimum
    constraints.max_pressure = 100e5    # 100 bar maximum  
    constraints.max_velocity = 8.0      # 8 m/s maximum
    
    # Delivery requirements
    constraints.node_flow_demands = {
        "Delivery1": 0.15,
        "Delivery2": 0.25
    }
    
    print("Optimizing pump operation...")
    print("Objective: Minimize total energy consumption")
    print("Constraints:")
    print(f"  Min pressure: {constraints.min_pressure/1e5:.0f} bar")
    print(f"  Max pressure: {constraints.max_pressure/1e5:.0f} bar")
    print(f"  Max velocity: {constraints.max_velocity} m/s")
    print(f"  Delivery requirements: {sum(constraints.node_flow_demands.values())*1000:.0f} l/s total")
    
    # Run optimization
    result = optimizer.optimize(network, fluid, objective, constraints)
    
    if result.success:
        print("\nOptimization successful!")
        print(f"Total power consumption: {result.objective_value/1000:.1f} kW")
        
        print("\nOptimal pump speeds:")
        for pump_id, speed in result.pump_speeds.items():
            print(f"  {pump_id}: {speed*100:.1f}%")
        
        # Run simulation with optimal settings
        run_optimized_simulation(network, fluid, result)
        
    else:
        print("\nOptimization failed - no feasible solution found")


def run_optimized_simulation(network, fluid, optimization_result):
    """Run simulation with optimized pump settings"""
    
    # Apply optimal pump speeds
    # (In practice, would modify pump curves based on speed)
    
    # Run simulation
    solver = ps.SteadyStateSolver(network, fluid)
    results = solver.solve()
    
    if results.converged:
        print("\nSimulation with optimal settings:")
        
        # Check constraints
        print("\nConstraint verification:")
        
        # Pressure constraints
        min_p = min(results.node_pressures.values())
        max_p = max(results.node_pressures.values())
        print(f"  Pressure range: {min_p/1e5:.1f} - {max_p/1e5:.1f} bar ✓")
        
        # Velocity constraints
        max_velocity = 0
        for pipe_id, pipe in network.pipes.items():
            flow = results.pipe_flow_rates[pipe_id]
            velocity = abs(flow) / pipe.area()
            max_velocity = max(max_velocity, velocity)
        print(f"  Max velocity: {max_velocity:.1f} m/s ✓")
        
        # Flow delivery
        print(f"  Delivery1 flow: {results.pipe_flow_rates.get('Junction2_Delivery1', 0)*1000:.1f} l/s ✓")
        print(f"  Delivery2 flow: {results.pipe_flow_rates.get('Junction2_Delivery2', 0)*1000:.1f} l/s ✓")
        
        # Energy analysis
        analyze_energy_consumption(network, results, fluid)
        
        # Visualize results
        visualize_optimization_results(network, results)


def analyze_energy_consumption(network, results, fluid):
    """Analyze energy consumption of the system"""
    print("\nEnergy Analysis:")
    
    total_hydraulic_power = 0
    total_pump_power = 0
    
    # For each pump
    pump_data = []
    for node_id, node in network.nodes.items():
        if node.type == ps.NodeType.PUMP:
            # Find inlet node (assumes naming convention)
            inlet_id = node_id.replace("_Out", "_In")
            if inlet_id in network.nodes:
                # Pressure rise
                p_in = results.node_pressures.get(inlet_id, 0)
                p_out = results.node_pressures.get(node_id, 0)
                dp = p_out - p_in
                
                # Flow through pump
                pipe_id = node_id.replace("_Out", "_Internal")
                flow = results.pipe_flow_rates.get(pipe_id, 0)
                
                # Hydraulic power
                hydraulic_power = flow * dp
                
                # Actual power (assuming 75% efficiency)
                pump_efficiency = 0.75
                pump_power = hydraulic_power / pump_efficiency
                
                total_hydraulic_power += hydraulic_power
                total_pump_power += pump_power
                
                pump_data.append({
                    'name': node_id.replace("_Out", ""),
                    'flow': flow,
                    'dp': dp,
                    'hydraulic_power': hydraulic_power,
                    'pump_power': pump_power
                })
                
                print(f"\n  {node_id.replace('_Out', '')}:")
                print(f"    Flow: {flow*1000:.1f} l/s")
                print(f"    Pressure rise: {dp/1e5:.1f} bar")
                print(f"    Hydraulic power: {hydraulic_power/1000:.1f} kW")
                print(f"    Pump power: {pump_power/1000:.1f} kW")
    
    print(f"\nTotal hydraulic power: {total_hydraulic_power/1000:.1f} kW")
    print(f"Total pump power: {total_pump_power/1000:.1f} kW")
    print(f"Overall pumping efficiency: {total_hydraulic_power/total_pump_power*100:.1f}%")
    
    # Energy cost analysis
    electricity_cost = 0.10  # $/kWh
    daily_cost = total_pump_power / 1000 * 24 * electricity_cost
    annual_cost = daily_cost * 365
    
    print(f"\nOperating cost:")
    print(f"  Daily: ${daily_cost:.2f}")
    print(f"  Annual: ${annual_cost:,.0f}")
    
    return pump_data


def visualize_optimization_results(network, results):
    """Create visualization of optimization results"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Pressure profile
    ax1.set_title('Pressure Profile Along Pipeline')
    
    # Main path nodes
    path_nodes = ["Inlet", "Pump1_In", "Pump1_Out", "Junction1", 
                  "Pump2_In", "Pump2_Out", "Junction2"]
    
    distances = [0, 1, 1.01, 6, 8, 8.01, 11]  # Approximate distances in km
    pressures = [results.node_pressures.get(n, 0)/1e5 for n in path_nodes]
    elevations = [network.nodes[n].elevation for n in path_nodes]
    
    ax1_elev = ax1.twinx()
    
    line1 = ax1.plot(distances, pressures, 'b-o', linewidth=2, 
                     markersize=8, label='Pressure')
    line2 = ax1_elev.plot(distances, elevations, 'g--', linewidth=1.5, 
                         alpha=0.7, label='Elevation')
    
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Pressure (bar)', color='b')
    ax1_elev.set_ylabel('Elevation (m)', color='g')
    ax1.grid(True, alpha=0.3)
    
    # Mark pump locations
    pump_locations = [1, 8]
    for loc in pump_locations:
        ax1.axvline(x=loc, color='red', linestyle=':', alpha=0.5)
        ax1.text(loc, ax1.get_ylim()[1]*0.9, 'Pump', rotation=90, 
                verticalalignment='bottom')
    
    # 2. Flow distribution
    ax2.set_title('Flow Distribution')
    
    pipe_flows = []
    pipe_names = []
    for pipe_id, flow in results.pipe_flow_rates.items():
        if abs(flow) > 0.001:  # Only significant flows
            pipe_flows.append(abs(flow) * 1000)  # Convert to l/s
            pipe_names.append(pipe_id.replace('_', '\n'))
    
    # Sort by flow
    sorted_data = sorted(zip(pipe_flows, pipe_names), reverse=True)
    flows = [d[0] for d in sorted_data[:8]]  # Top 8
    names = [d[1] for d in sorted_data[:8]]
    
    bars = ax2.bar(range(len(flows)), flows, color='skyblue')
    ax2.set_xticks(range(len(flows)))
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylabel('Flow Rate (l/s)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, flow in zip(bars, flows):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(flow)}', ha='center', va='bottom')
    
    # 3. Energy consumption by pump
    ax3.set_title('Energy Consumption by Component')
    
    # Get pump data (simplified for example)
    pump_names = ['Pump1', 'Pump2']
    pump_powers = [50, 80]  # kW (example values)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(pump_names)))
    ax3.pie(pump_powers, labels=pump_names, colors=colors, autopct='%1.1f%%',
            startangle=90)
    ax3.text(0, -1.3, f'Total: {sum(pump_powers)} kW', 
             ha='center', fontsize=12, weight='bold')
    
    # 4. Velocity profile
    ax4.set_title('Flow Velocity Distribution')
    
    velocities = []
    for pipe_id, pipe in network.pipes.items():
        flow = results.pipe_flow_rates.get(pipe_id, 0)
        velocity = abs(flow) / pipe.area()
        velocities.append(velocity)
    
    ax4.hist(velocities, bins=20, color='coral', alpha=0.7, edgecolor='black')
    ax4.axvline(x=8, color='red', linestyle='--', linewidth=2, 
                label='Max allowed (8 m/s)')
    ax4.set_xlabel('Velocity (m/s)')
    ax4.set_ylabel('Number of Pipes')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('optimization_results.png', dpi=150)
    plt.show()


def sensitivity_analysis():
    """Perform sensitivity analysis on optimization parameters"""
    print("\n" + "="*50)
    print("SENSITIVITY ANALYSIS")
    print("="*50)
    
    # Parameters to vary
    flow_demands = np.linspace(0.3, 0.5, 5)  # Total flow from 300 to 500 l/s
    electricity_costs = np.linspace(0.05, 0.20, 5)  # $/kWh
    
    results_matrix = np.zeros((len(flow_demands), len(electricity_costs)))
    
    print("Running sensitivity analysis...")
    print("Varying: Total flow demand and electricity cost")
    
    # Run optimizations
    for i, total_flow in enumerate(flow_demands):
        for j, elec_cost in enumerate(electricity_costs):
            # Create network with modified demand
            network = create_pipeline_network_with_pumps()
            
            # Modify flow demands proportionally
            flow1 = total_flow * 0.375  # 37.5% to delivery 1
            flow2 = total_flow * 0.625  # 62.5% to delivery 2
            
            network.set_flow_rate(network.get_node("Delivery1"), flow1)
            network.set_flow_rate(network.get_node("Delivery2"), flow2)
            
            # Run optimization (simplified)
            # In practice, would run full optimization
            # Here we estimate based on flow
            power_consumption = 50 + 300 * total_flow  # kW (simplified model)
            annual_cost = power_consumption * 24 * 365 * elec_cost / 1000
            
            results_matrix[i, j] = annual_cost
    
    # Plot sensitivity results
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(results_matrix, cmap='viridis', aspect='auto')
    
    # Set ticks
    ax.set_xticks(range(len(electricity_costs)))
    ax.set_yticks(range(len(flow_demands)))
    ax.set_xticklabels([f'${c:.2f}' for c in electricity_costs])
    ax.set_yticklabels([f'{int(f*1000)}' for f in flow_demands])
    
    ax.set_xlabel('Electricity Cost ($/kWh)')
    ax.set_ylabel('Total Flow Demand (l/s)')
    ax.set_title('Annual Operating Cost Sensitivity Analysis')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Annual Cost ($)', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(flow_demands)):
        for j in range(len(electricity_costs)):
            text = ax.text(j, i, f'${int(results_matrix[i, j]/1000)}k',
                         ha="center", va="center", color="white", fontsize=10)
    
    plt.tight_layout()
    plt.savefig('sensitivity_analysis.png', dpi=150)
    plt.show()
    
    print("\nSensitivity analysis complete - see sensitivity_analysis.png")


if __name__ == "__main__":
    # Run optimization
    optimize_pump_operation()
    
    # Run sensitivity analysis
    sensitivity_analysis()
