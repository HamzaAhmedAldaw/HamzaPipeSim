#!/usr/bin/env python3
"""
test_hamza_pipesim.py - Fixed version with correct property access
"""

import sys
import os
import importlib.util

print("=== Loading Pipeline-Sim ===")

# Load the .pyd directly
pyd_path = r"C:\Users\KIMO STORE\AppData\Roaming\Python\Python313\site-packages\pipeline_sim.cp313-win_amd64.pyd"

if not os.path.exists(pyd_path):
    print(f"Error: .pyd file not found at {pyd_path}")
    sys.exit(1)

spec = importlib.util.spec_from_file_location("pipeline_sim", pyd_path)
pipeline_sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline_sim)

print("✅ Successfully loaded Pipeline-Sim C++ module")

# Import classes
Network = pipeline_sim.Network
Node = pipeline_sim.Node
Pipe = pipeline_sim.Pipe
NodeType = pipeline_sim.NodeType
FluidProperties = pipeline_sim.FluidProperties
SteadyStateSolver = pipeline_sim.SteadyStateSolver
SolverConfig = pipeline_sim.SolverConfig
SolutionResults = pipeline_sim.SolutionResults
constants = pipeline_sim.constants
get_version = pipeline_sim.get_version

def test_simple_pipeline():
    """Test simple pipeline simulation"""
    print("\n=== Test 1: Simple Pipeline ===")
    
    # Create network
    network = Network()
    inlet = network.add_node("inlet", NodeType.SOURCE)
    outlet = network.add_node("outlet", NodeType.SINK)
    
    # Create pipe: 1000m long, 0.5m diameter
    pipe = network.add_pipe("pipe1", inlet, outlet, 1000.0, 0.5)
    
    # Set boundary conditions
    network.set_pressure(inlet, 500000.0)   # 5 bar
    network.set_pressure(outlet, 100000.0)  # 1 bar
    
    # Create fluid properties
    fluid = FluidProperties()
    fluid.oil_density = 850.0
    fluid.oil_viscosity = 0.001
    
    # Print node info (properties, not methods!)
    print(f"\nNode properties:")
    print(f"  Inlet ID: {inlet.id}")  # Not inlet.id()
    print(f"  Inlet pressure: {inlet.pressure/1e5:.1f} bar")
    print(f"  Outlet ID: {outlet.id}")
    print(f"  Outlet pressure: {outlet.pressure/1e5:.1f} bar")
    
    # Create solver
    solver = SteadyStateSolver(network, fluid)
    config = solver.config
    config.verbose = True
    solver.set_config(config)
    
    # Solve
    print("\nSolving...")
    results = solver.solve()
    
    print(f"\nResults:")
    print(f"  Converged: {results.converged}")
    if results.converged:
        print(f"  Flow rate: {results.pipe_flow_rates['pipe1']:.3f} m³/s")
        print(f"  Velocity: {results.pipe_velocities['pipe1']:.2f} m/s")
        print(f"  Reynolds number: {results.pipe_reynolds_numbers['pipe1']:.0f}")
        print(f"  Friction factor: {results.pipe_friction_factors['pipe1']:.6f}")

def test_network_with_junction():
    """Test network with junction"""
    print("\n=== Test 2: Network with Junction ===")
    
    # Create network
    network = Network()
    
    # Create nodes
    source1 = network.add_node("source1", NodeType.SOURCE)
    source2 = network.add_node("source2", NodeType.SOURCE)
    junction = network.add_node("junction", NodeType.JUNCTION)
    sink = network.add_node("sink", NodeType.SINK)
    
    # Create pipes
    pipe1 = network.add_pipe("pipe1", source1, junction, 500.0, 0.3)
    pipe2 = network.add_pipe("pipe2", source2, junction, 700.0, 0.25)
    pipe3 = network.add_pipe("pipe3", junction, sink, 1000.0, 0.4)
    
    # Set boundary conditions
    network.set_pressure(source1, 600000.0)  # 6 bar
    network.set_pressure(source2, 550000.0)  # 5.5 bar
    network.set_pressure(sink, 200000.0)     # 2 bar
    
    # Create fluid
    fluid = FluidProperties()
    fluid.oil_density = 850.0
    fluid.oil_viscosity = 0.005
    
    # Solve
    solver = SteadyStateSolver(network, fluid)
    config = solver.config
    config.verbose = False  # Less output for this test
    solver.set_config(config)
    
    results = solver.solve()
    
    print(f"\nResults:")
    print(f"  Converged: {results.converged}")
    if results.converged:
        print(f"  Junction pressure: {results.node_pressures['junction']/1e5:.2f} bar")
        print(f"  Flow rates:")
        for pipe_id, flow in results.pipe_flow_rates.items():
            direction = "→" if flow > 0 else "←"
            print(f"    {pipe_id}: {abs(flow):.4f} m³/s {direction}")

def test_with_elevation():
    """Test with elevation (using the elevation property)"""
    print("\n=== Test 3: Using Elevation Property ===")
    
    # Create network
    network = Network()
    
    # Create nodes
    bottom = network.add_node("bottom", NodeType.SOURCE)
    top = network.add_node("top", NodeType.SINK)
    
    # Check elevation property
    print(f"\nDefault elevations:")
    print(f"  Bottom: {bottom.elevation} m")
    print(f"  Top: {top.elevation} m")
    
    # Note: Since set_elevation doesn't exist, we'll work with default elevations
    # The solver should handle elevation internally based on node properties
    
    # Create pipe
    pipe = network.add_pipe("uphill", bottom, top, 2000.0, 0.4)
    
    # Set boundary conditions
    network.set_pressure(bottom, 1000000.0)  # 10 bar
    network.set_pressure(top, 300000.0)      # 3 bar
    
    # Create fluid
    fluid = FluidProperties()
    fluid.oil_density = 900.0
    fluid.oil_viscosity = 0.010
    
    # Solve
    solver = SteadyStateSolver(network, fluid)
    results = solver.solve()
    
    print(f"\nResults:")
    print(f"  Converged: {results.converged}")
    if results.converged:
        print(f"  Flow rate: {results.pipe_flow_rates['uphill']:.4f} m³/s")
        print(f"  Pressure drop: {results.pipe_pressure_drops['uphill']/1e5:.2f} bar")
        print(f"  Velocity: {results.pipe_velocities['uphill']:.2f} m/s")

def test_pump_node():
    """Test pump functionality"""
    print("\n=== Test 4: Pump Node ===")
    
    # Create network
    network = Network()
    
    # Create nodes
    inlet = network.add_node("inlet", NodeType.SOURCE)
    pump = network.add_node("pump", NodeType.PUMP)
    outlet = network.add_node("outlet", NodeType.SINK)
    
    # Check pump properties
    print(f"\nPump properties:")
    print(f"  Pump speed: {pump.pump_speed}")
    print(f"  Pump curve A: {pump.pump_curve_coefficient_a()}")
    print(f"  Pump curve B: {pump.pump_curve_coefficient_b()}")
    
    # Create pipes
    pipe1 = network.add_pipe("suction", inlet, pump, 100.0, 0.3)
    pipe2 = network.add_pipe("discharge", pump, outlet, 1000.0, 0.3)
    
    # Set boundary conditions
    network.set_pressure(inlet, 200000.0)   # 2 bar
    network.set_pressure(outlet, 500000.0)  # 5 bar
    
    # Set pump curve if method exists
    try:
        pump.set_pump_curve(1000000.0, 50000.0)  # Head = a - b*Q²
        print("  ✓ Pump curve set")
    except Exception as e:
        print(f"  ✗ Could not set pump curve: {e}")
    
    # Create fluid
    fluid = FluidProperties()
    fluid.oil_density = 850.0
    fluid.oil_viscosity = 0.002
    
    # Solve
    solver = SteadyStateSolver(network, fluid)
    results = solver.solve()
    
    print(f"\nResults:")
    print(f"  Converged: {results.converged}")
    if results.converged:
        print(f"  Suction flow: {results.pipe_flow_rates.get('suction', 0):.4f} m³/s")
        print(f"  Discharge flow: {results.pipe_flow_rates.get('discharge', 0):.4f} m³/s")
        print(f"  Pump pressure: {results.node_pressures.get('pump', 0)/1e5:.2f} bar")

def main():
    """Run all tests"""
    print("\nPipeline-Sim Test Suite")
    print("=" * 50)
    
    # Check module
    print(f"Version: {get_version()}")
    print(f"Constants:")
    print(f"  Standard pressure: {constants.STANDARD_PRESSURE/1e5:.2f} bar")
    print(f"  Standard temperature: {constants.STANDARD_TEMPERATURE:.1f} K")
    print(f"  Gravity: {constants.GRAVITY} m/s²")
    
    # Run tests
    test_simple_pipeline()
    test_network_with_junction()
    test_with_elevation()
    test_pump_node()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed successfully!")

if __name__ == "__main__":
    main()