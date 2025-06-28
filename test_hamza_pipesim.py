{
  `command`: `create`,
  `path`: `pipeline_simulation_working.py`,
  `file_text`: `\"\"\"
Working Pipeline-Sim Demonstration
Fixed to use correct argument syntax for add_pipe
\"\"\"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os
import sys

# Module loading with fallback options
try:
    # Try standard import first
    import pipeline_sim as ps
    print(\"✓ Pipeline-Sim module loaded successfully\")
except ImportError:
    # Try loading from build directory
    try:
        pyd_path = r\"C:\\Users\\KIMO STORE\\HamzaPipeSim\\build\\lib.win-amd64-cpython-313\\pipeline_sim.cp313-win_amd64.pyd\"
        if os.path.exists(pyd_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location(\"pipeline_sim\", pyd_path)
            ps = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ps)
            print(f\"✓ Pipeline-Sim loaded from: {pyd_path}\")
        else:
            raise ImportError(\"PYD file not found\")
    except:
        print(\"✗ Failed to load Pipeline-Sim module\")
        sys.exit(1)


def run_simple_example():
    \"\"\"Run a simple example to test basic functionality\"\"\"
    print(\"\
\" + \"=\"*60)
    print(\"SIMPLE PIPELINE EXAMPLE\")
    print(\"=\"*60)
    
    # Create network
    network = ps.Network()
    
    # Add nodes
    inlet = network.add_node(\"inlet\", ps.NodeType.SOURCE)
    outlet = network.add_node(\"outlet\", ps.NodeType.SINK)
    
    # Set elevations using property syntax
    inlet.elevation = 0
    outlet.elevation = 100
    
    # Add pipe - using POSITIONAL arguments, not keyword arguments!
    pipe = network.add_pipe(\"main_pipe\", inlet, outlet, 1000.0, 0.3)
    pipe.roughness = 0.000045
    
    # Set boundary conditions
    network.set_pressure(inlet, 50e5)  # 50 bar
    network.set_pressure(outlet, 20e5)  # 20 bar
    
    # Create fluid
    fluid = ps.FluidProperties()
    fluid.gas_density = 0.75
    fluid.gas_viscosity = 1.2e-5
    fluid.gas_fraction = 1.0
    fluid.oil_fraction = 0.0
    fluid.water_fraction = 0.0
    
    print(f\"Fluid mixture density: {fluid.mixture_density():.2f} kg/m³\")
    print(f\"Fluid mixture viscosity: {fluid.mixture_viscosity()*1000:.3f} cP\")
    
    # Solve
    solver = ps.SteadyStateSolver(network, fluid)
    results = solver.solve()
    
    print(f\"\
Simple Example Results:\")
    print(f\"Converged: {results.converged}\")
    if results.converged:
        print(f\"Iterations: {results.iterations}\")
        print(f\"Flow rate: {results.pipe_flow_rates['main_pipe']:.6f} m³/s\")
        print(f\"Pressure drop: {results.pipe_pressure_drops['main_pipe']/1e5:.2f} bar\")
        print(f\"Inlet pressure: {results.node_pressures['inlet']/1e5:.2f} bar\")
        print(f\"Outlet pressure: {results.node_pressures['outlet']/1e5:.2f} bar\")


def create_three_node_network():
    \"\"\"Create a simple three-node network for testing\"\"\"
    print(\"\
\" + \"=\"*60)
    print(\"THREE-NODE NETWORK EXAMPLE\")
    print(\"=\"*60)
    
    network = ps.Network()
    
    # Create nodes
    source = network.add_node(\"Source\", ps.NodeType.SOURCE)
    junction = network.add_node(\"Junction\", ps.NodeType.JUNCTION)
    sink = network.add_node(\"Sink\", ps.NodeType.SINK)
    
    # Set elevations
    source.elevation = 0
    junction.elevation = 50
    sink.elevation = 30
    
    # Create pipes - POSITIONAL arguments
    pipe1 = network.add_pipe(\"Pipe-1\", source, junction, 1000.0, 0.25)
    pipe2 = network.add_pipe(\"Pipe-2\", junction, sink, 1500.0, 0.3)
    
    # Set pipe properties
    pipe1.roughness = 0.000045
    pipe2.roughness = 0.000045
    
    # Set boundary conditions
    network.set_pressure(source, 70e5)  # 70 bar
    network.set_pressure(sink, 30e5)    # 30 bar
    
    return network


def create_gas_pipeline_network():
    \"\"\"Create a realistic gas pipeline network\"\"\"
    print(\"\
\" + \"=\"*60)
    print(\"CREATING GAS PIPELINE NETWORK\")
    print(\"=\"*60)
    
    network = ps.Network()
    
    # Gas wells (sources)
    well1 = network.add_node(\"Well-1\", ps.NodeType.SOURCE)
    well1.elevation = -500
    
    well2 = network.add_node(\"Well-2\", ps.NodeType.SOURCE)
    well2.elevation = -450
    
    well3 = network.add_node(\"Well-3\", ps.NodeType.SOURCE)
    well3.elevation = -480
    
    # Surface facilities
    manifold = network.add_node(\"Manifold\", ps.NodeType.JUNCTION)
    manifold.elevation = 0
    
    # Compression station
    compressor = network.add_node(\"Compressor\", ps.NodeType.COMPRESSOR)
    compressor.elevation = 5
    
    # Processing plant
    processing = network.add_node(\"Processing\", ps.NodeType.JUNCTION)
    processing.elevation = 10
    
    # Storage
    storage = network.add_node(\"Storage\", ps.NodeType.SINK)
    storage.elevation = 15
    
    # Connect wells to manifold - POSITIONAL arguments: id, upstream, downstream, length, diameter
    pipe1 = network.add_pipe(\"Flowline-1\", well1, manifold, 2000.0, 0.2)
    pipe1.roughness = 0.000045
    pipe1.inclination = np.arctan(500/2000)
    
    pipe2 = network.add_pipe(\"Flowline-2\", well2, manifold, 1800.0, 0.2)
    pipe2.roughness = 0.000045
    pipe2.inclination = np.arctan(450/1800)
    
    pipe3 = network.add_pipe(\"Flowline-3\", well3, manifold, 2100.0, 0.18)
    pipe3.roughness = 0.000045
    pipe3.inclination = np.arctan(480/2100)
    
    # Manifold to compressor
    trunk_line = network.add_pipe(\"Trunk-Line\", manifold, compressor, 5000.0, 0.4)
    trunk_line.roughness = 0.00003
    
    # Compressor to processing
    hp_line = network.add_pipe(\"HP-Line\", compressor, processing, 10000.0, 0.35)
    hp_line.roughness = 0.00003
    
    # Processing to storage
    storage_line = network.add_pipe(\"Storage-Line\", processing, storage, 1000.0, 0.3)
    storage_line.roughness = 0.000045
    
    print(f\"Network created with {len(network.nodes)} nodes and {len(network.pipes)} pipes\")
    
    # Set boundary conditions
    network.set_pressure(well1, 120e5)  # 120 bar
    network.set_pressure(well2, 118e5)  # 118 bar
    network.set_pressure(well3, 115e5)  # 115 bar
    network.set_pressure(storage, 30e5)  # 30 bar delivery pressure
    
    return network


def create_fluid_properties(fluid_type=\"gas\"):
    \"\"\"Create fluid properties for simulation\"\"\"
    print(f\"\
Creating {fluid_type} fluid properties...\")
    
    fluid = ps.FluidProperties()
    
    if fluid_type == \"gas\":
        # Natural gas properties
        fluid.oil_density = 1.0      # Minimal oil
        fluid.gas_density = 0.75     # Relative to air
        fluid.water_density = 1000.0 # Water
        
        fluid.oil_viscosity = 1e-3
        fluid.gas_viscosity = 1.2e-5
        fluid.water_viscosity = 1e-3
        
        fluid.gas_fraction = 0.99
        fluid.oil_fraction = 0.005
        fluid.water_fraction = 0.005
        
    elif fluid_type == \"oil\":
        # Crude oil properties
        fluid.oil_density = 850.0
        fluid.gas_density = 0.8
        fluid.water_density = 1025.0
        
        fluid.oil_viscosity = 0.01
        fluid.gas_viscosity = 1.5e-5
        fluid.water_viscosity = 0.001
        
        # Multiphase fractions
        fluid.water_cut = 0.2
        fluid.gas_oil_ratio = 100
        fluid.water_fraction = 0.2
        fluid.oil_fraction = 0.7
        fluid.gas_fraction = 0.1
    
    print(f\"Mixture density: {fluid.mixture_density():.2f} kg/m³\")
    print(f\"Mixture viscosity: {fluid.mixture_viscosity()*1000:.3f} cP\")
    
    return fluid


def analyze_results(network, results):
    \"\"\"Analyze simulation results\"\"\"
    if not results or not results.converged:
        print(\"No valid results to analyze\")
        return
        
    print(\"\
\" + \"=\"*60)
    print(\"SIMULATION RESULTS ANALYSIS\")
    print(\"=\"*60)
    
    # Node pressure analysis
    print(\"\
1. Node Pressures:\")
    print(\"-\" * 50)
    print(f\"{'Node':<15} {'Pressure (bar)':<15} {'Elevation (m)':<15}\")
    print(\"-\" * 50)
    
    for node_id, pressure in results.node_pressures.items():
        node = network.get_node(node_id)
        if node:
            print(f\"{node_id:<15} {pressure/1e5:>14.2f} {node.elevation:>14.2f}\")
    
    # Flow rate analysis
    print(\"\
2. Pipe Flow Rates:\")
    print(\"-\" * 50)
    print(f\"{'Pipe':<15} {'Flow (m³/s)':<15} {'Velocity (m/s)':<15}\")
    print(\"-\" * 50)
    
    for pipe_id, flow in results.pipe_flow_rates.items():
        pipe = network.get_pipe(pipe_id)
        if pipe:
            velocity = flow / pipe.area()
            print(f\"{pipe_id:<15} {flow:>14.4f} {velocity:>14.2f}\")
    
    # Pressure drop analysis
    print(\"\
3. Pressure Drops:\")
    print(\"-\" * 50)
    print(f\"{'Pipe':<15} {'ΔP (bar)':<15} {'ΔP/L (bar/km)':<15}\")
    print(\"-\" * 50)
    
    for pipe_id, dp in results.pipe_pressure_drops.items():
        pipe = network.get_pipe(pipe_id)
        if pipe:
            dp_per_km = (dp/1e5) / (pipe.length/1000)
            print(f\"{pipe_id:<15} {dp/1e5:>14.3f} {dp_per_km:>14.3f}\")


def visualize_results(network, results):
    \"\"\"Create visualization of results\"\"\"
    if not results or not results.converged:
        return
        
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Pressure distribution
    nodes = list(results.node_pressures.keys())
    pressures = [results.node_pressures[n]/1e5 for n in nodes]
    
    ax1.bar(range(len(nodes)), pressures, color='skyblue')
    ax1.set_xticks(range(len(nodes)))
    ax1.set_xticklabels(nodes, rotation=45, ha='right')
    ax1.set_ylabel('Pressure (bar)')
    ax1.set_title('Node Pressures')
    ax1.grid(True, alpha=0.3)
    
    # 2. Flow velocities
    pipes = list(results.pipe_flow_rates.keys())
    velocities = []
    for pipe_id in pipes:
        pipe = network.get_pipe(pipe_id)
        if pipe:
            flow = results.pipe_flow_rates[pipe_id]
            velocity = abs(flow) / pipe.area()
            velocities.append(velocity)
    
    bars = ax2.bar(range(len(pipes)), velocities)
    
    # Color code velocities
    for bar, vel in zip(bars, velocities):
        if vel > 20:
            bar.set_color('red')
        elif vel > 15:
            bar.set_color('orange')
        else:
            bar.set_color('green')
    
    ax2.set_xticks(range(len(pipes)))
    ax2.set_xticklabels(pipes, rotation=45, ha='right')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Flow Velocities')
    ax2.axhline(y=20, color='r', linestyle='--', alpha=0.5, label='Max limit')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Pressure drops
    drops = [results.pipe_pressure_drops[p]/1e5 for p in pipes]
    
    ax3.bar(range(len(pipes)), drops, color='coral')
    ax3.set_xticks(range(len(pipes)))
    ax3.set_xticklabels(pipes, rotation=45, ha='right')
    ax3.set_ylabel('Pressure Drop (bar)')
    ax3.set_title('Pressure Drops')
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary statistics
    ax4.axis('off')
    summary_text = f\"\"\"Simulation Summary
    -----------------
    Converged: Yes
    Iterations: {results.iterations}
    Residual: {results.residual:.2e}
    Time: {results.computation_time:.3f} s
    
    Network Statistics
    -----------------
    Total Nodes: {len(network.nodes)}
    Total Pipes: {len(network.pipes)}
    
    Flow Summary
    -----------
    Total Flow: {sum(results.pipe_flow_rates.values()):.4f} m³/s
    Avg Velocity: {np.mean(velocities):.2f} m/s
    Max Velocity: {max(velocities):.2f} m/s
    Total ΔP: {sum(results.pipe_pressure_drops.values())/1e5:.2f} bar\"\"\"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('pipeline_results.png', dpi=150, bbox_inches='tight')
    plt.show()


def save_results(network, results):
    \"\"\"Save results to CSV files\"\"\"
    if not results or not results.converged:
        return
        
    # Node results
    node_data = []
    for node_id, pressure in results.node_pressures.items():
        node = network.get_node(node_id)
        if node:
            node_data.append({
                'Node ID': node_id,
                'Pressure (bar)': pressure/1e5,
                'Temperature (K)': results.node_temperatures.get(node_id, 288.15),
                'Elevation (m)': node.elevation
            })
    
    if node_data:
        pd.DataFrame(node_data).to_csv('node_results.csv', index=False)
        print(\"✓ Node results saved to node_results.csv\")
    
    # Pipe results
    pipe_data = []
    for pipe_id, flow in results.pipe_flow_rates.items():
        pipe = network.get_pipe(pipe_id)
        if pipe:
            pipe_data.append({
                'Pipe ID': pipe_id,
                'Flow Rate (m³/s)': flow,
                'Velocity (m/s)': flow / pipe.area(),
                'Pressure Drop (bar)': results.pipe_pressure_drops.get(pipe_id, 0) / 1e5,
                'Length (m)': pipe.length,
                'Diameter (m)': pipe.diameter
            })
    
    if pipe_data:
        pd.DataFrame(pipe_data).to_csv('pipe_results.csv', index=False)
        print(\"✓ Pipe results saved to pipe_results.csv\")


def main():
    \"\"\"Main simulation workflow\"\"\"
    print(\"=\"*60)
    print(\"PIPELINE-SIM WORKING DEMONSTRATION\")
    print(\"=\"*60)
    
    try:
        # 1. Run simple example first
        run_simple_example()
        
        # 2. Run three-node network
        print(\"\
Running three-node network simulation...\")
        network = create_three_node_network()
        fluid = create_fluid_properties(\"gas\")
        
        solver = ps.SteadyStateSolver(network, fluid)
        results = solver.solve()
        
        if results.converged:
            print(f\"✓ Three-node simulation converged in {results.iterations} iterations\")
            analyze_results(network, results)
        
        # 3. Run full gas pipeline network
        print(\"\
Running full gas pipeline network simulation...\")
        network = create_gas_pipeline_network()
        fluid = create_fluid_properties(\"gas\")
        
        solver = ps.SteadyStateSolver(network, fluid)
        results = solver.solve()
        
        if results.converged:
            print(f\"\
✓ Full network simulation converged in {results.iterations} iterations\")
            print(f\"  Residual: {results.residual:.2e}\")
            print(f\"  Time: {results.computation_time:.3f} seconds\")
            
            # Analyze and visualize
            analyze_results(network, results)
            visualize_results(network, results)
            save_results(network, results)
        else:
            print(\"\
✗ Full network simulation failed to converge\")
            
    except Exception as e:
        print(f\"\
Error: {e}\")
        import traceback
        traceback.print_exc()
    
    print(\"\
\" + \"=\"*60)
    print(\"DEMONSTRATION COMPLETE\")
    print(\"=\"*60)


if __name__ == \"__main__\":
    main()
`
}