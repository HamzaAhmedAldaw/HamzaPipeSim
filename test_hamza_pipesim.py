"""
Simple Pipeline Test
Tests the most basic pipeline configuration possible
"""

import sys
import os
import numpy as np

# Load the module
try:
    import pipeline_sim as ps
    print("✓ Pipeline-Sim module loaded successfully")
except ImportError:
    # Try loading from build directory
    pyd_path = r"C:\Users\KIMO STORE\HamzaPipeSim\build\lib.win-amd64-cpython-313\pipeline_sim.cp313-win_amd64.pyd"
    if os.path.exists(pyd_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("pipeline_sim", pyd_path)
        ps = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ps)
        print(f"✓ Pipeline-Sim loaded from: {pyd_path}")
    else:
        print("✗ Failed to load Pipeline-Sim module")
        sys.exit(1)


def test_single_pipe():
    """Test 1: Single horizontal pipe"""
    print("\n" + "="*60)
    print("TEST 1: SINGLE HORIZONTAL PIPE")
    print("="*60)
    
    # Create network
    network = ps.Network()
    
    # Create two nodes
    inlet = network.add_node("INLET", ps.NodeType.SOURCE)
    outlet = network.add_node("OUTLET", ps.NodeType.SINK)
    
    # Set elevations (horizontal pipe)
    inlet.elevation = 0.0
    outlet.elevation = 0.0
    
    # Create pipe: 1000m long, 0.3m diameter
    pipe = network.add_pipe("PIPE-1", inlet, outlet, 1000.0, 0.3)
    pipe.roughness = 0.000045  # Commercial steel
    
    # Create fluid (water for simplicity)
    fluid = ps.FluidProperties()
    fluid.oil_density = 1000.0    # Water density
    fluid.gas_density = 1.2       # Air (not used)
    fluid.water_density = 1000.0  # Water
    
    fluid.oil_viscosity = 0.001   # Water viscosity
    fluid.gas_viscosity = 0.00002
    fluid.water_viscosity = 0.001
    
    # Single phase water
    fluid.oil_fraction = 1.0  # Treat water as "oil" phase
    fluid.gas_fraction = 0.0
    fluid.water_fraction = 0.0
    
    print(f"Fluid density: {fluid.mixture_density():.1f} kg/m³")
    print(f"Fluid viscosity: {fluid.mixture_viscosity()*1000:.1f} cP")
    
    # Test Case 1A: Pressure-Pressure boundary conditions
    print("\nCase 1A: Pressure inlet (10 bar) - Pressure outlet (9 bar)")
    network.set_pressure(inlet, 10e5)   # 10 bar
    network.set_pressure(outlet, 9e5)   # 9 bar
    
    # Solve
    solver = ps.SteadyStateSolver(network, fluid)
    results = solver.solve()
    
    print(f"Converged: {results.converged}")
    if results.converged:
        print(f"Flow rate: {results.pipe_flow_rates['PIPE-1']:.4f} m³/s")
        print(f"Pressure drop: {(results.node_pressures['INLET'] - results.node_pressures['OUTLET'])/1e5:.2f} bar")
        
        # Check with analytical solution
        # For laminar flow: Q = π*ΔP*D⁴/(128*μ*L)
        # For turbulent: use Darcy-Weisbach
        flow = results.pipe_flow_rates['PIPE-1']
        velocity = flow / (np.pi * 0.15**2)
        reynolds = 1000 * velocity * 0.3 / 0.001
        print(f"Velocity: {velocity:.2f} m/s")
        print(f"Reynolds: {reynolds:.0f}")


def test_three_node_network():
    """Test 2: Three node network (branching)"""
    print("\n" + "="*60)
    print("TEST 2: THREE NODE BRANCHING NETWORK")
    print("="*60)
    
    # Create network
    network = ps.Network()
    
    # Create nodes
    source = network.add_node("SOURCE", ps.NodeType.SOURCE)
    junction = network.add_node("JUNCTION", ps.NodeType.JUNCTION)
    sink1 = network.add_node("SINK1", ps.NodeType.SINK)
    sink2 = network.add_node("SINK2", ps.NodeType.SINK)
    
    # All at same elevation
    source.elevation = 0.0
    junction.elevation = 0.0
    sink1.elevation = 0.0
    sink2.elevation = 0.0
    
    # Create pipes
    pipe1 = network.add_pipe("SUPPLY", source, junction, 1000.0, 0.4)
    pipe2 = network.add_pipe("BRANCH1", junction, sink1, 500.0, 0.3)
    pipe3 = network.add_pipe("BRANCH2", junction, sink2, 500.0, 0.3)
    
    # Set roughness
    for pipe in [pipe1, pipe2, pipe3]:
        pipe.roughness = 0.000045
    
    # Same fluid as before
    fluid = ps.FluidProperties()
    fluid.oil_density = 1000.0
    fluid.gas_density = 1.2
    fluid.water_density = 1000.0
    fluid.oil_viscosity = 0.001
    fluid.gas_viscosity = 0.00002
    fluid.water_viscosity = 0.001
    fluid.oil_fraction = 1.0
    fluid.gas_fraction = 0.0
    fluid.water_fraction = 0.0
    
    # Set boundary conditions
    network.set_pressure(source, 10e5)    # 10 bar
    network.set_pressure(sink1, 8e5)      # 8 bar
    network.set_pressure(sink2, 8e5)      # 8 bar
    
    print("Network configuration:")
    print("  SOURCE (10 bar) --> JUNCTION --> SINK1 (8 bar)")
    print("                              +--> SINK2 (8 bar)")
    
    # Solve
    solver = ps.SteadyStateSolver(network, fluid)
    results = solver.solve()
    
    print(f"\nConverged: {results.converged}")
    if results.converged:
        print("\nResults:")
        print(f"Supply flow: {results.pipe_flow_rates['SUPPLY']:.4f} m³/s")
        print(f"Branch 1 flow: {results.pipe_flow_rates['BRANCH1']:.4f} m³/s")
        print(f"Branch 2 flow: {results.pipe_flow_rates['BRANCH2']:.4f} m³/s")
        print(f"Junction pressure: {results.node_pressures['JUNCTION']/1e5:.2f} bar")
        
        # Check mass balance
        supply = results.pipe_flow_rates['SUPPLY']
        branch1 = results.pipe_flow_rates['BRANCH1']
        branch2 = results.pipe_flow_rates['BRANCH2']
        imbalance = abs(supply - branch1 - branch2)
        print(f"\nMass balance check: {imbalance:.2e} m³/s")


def test_elevation_change():
    """Test 3: Pipe with elevation change"""
    print("\n" + "="*60)
    print("TEST 3: VERTICAL PIPE (ELEVATION CHANGE)")
    print("="*60)
    
    # Create network
    network = ps.Network()
    
    # Create nodes
    bottom = network.add_node("BOTTOM", ps.NodeType.SOURCE)
    top = network.add_node("TOP", ps.NodeType.SINK)
    
    # Vertical pipe - 100m elevation change
    bottom.elevation = 0.0
    top.elevation = 100.0
    
    # Create pipe
    pipe = network.add_pipe("RISER", bottom, top, 100.0, 0.2)
    pipe.roughness = 0.000045
    
    # Water properties
    fluid = ps.FluidProperties()
    fluid.oil_density = 1000.0
    fluid.gas_density = 1.2
    fluid.water_density = 1000.0
    fluid.oil_viscosity = 0.001
    fluid.gas_viscosity = 0.00002
    fluid.water_viscosity = 0.001
    fluid.oil_fraction = 1.0
    fluid.gas_fraction = 0.0
    fluid.water_fraction = 0.0
    
    # Set pressures - need to overcome hydrostatic head
    # Hydrostatic pressure = ρ*g*h = 1000*9.81*100 = 981000 Pa = 9.81 bar
    network.set_pressure(bottom, 20e5)  # 20 bar
    network.set_pressure(top, 5e5)      # 5 bar
    
    print("Configuration: 100m vertical riser")
    print(f"Bottom pressure: 20 bar")
    print(f"Top pressure: 5 bar")
    print(f"Hydrostatic head: {1000*9.81*100/1e5:.2f} bar")
    
    # Solve
    solver = ps.SteadyStateSolver(network, fluid)
    results = solver.solve()
    
    print(f"\nConverged: {results.converged}")
    if results.converged:
        flow = results.pipe_flow_rates['RISER']
        dp = results.pipe_pressure_drops['RISER']
        print(f"Flow rate: {flow:.4f} m³/s")
        print(f"Total pressure drop: {dp/1e5:.2f} bar")
        print(f"Available driving pressure: {(20-5-9.81):.2f} bar")


def test_gas_flow():
    """Test 4: Gas flow in horizontal pipe"""
    print("\n" + "="*60)
    print("TEST 4: GAS FLOW")
    print("="*60)
    
    # Create network
    network = ps.Network()
    
    # Create nodes
    inlet = network.add_node("INLET", ps.NodeType.SOURCE)
    outlet = network.add_node("OUTLET", ps.NodeType.SINK)
    
    inlet.elevation = 0.0
    outlet.elevation = 0.0
    
    # Create pipe
    pipe = network.add_pipe("GAS-PIPE", inlet, outlet, 5000.0, 0.5)
    pipe.roughness = 0.00003  # Smooth pipe
    
    # Natural gas properties
    fluid = ps.FluidProperties()
    
    # Gas at 50 bar, 20°C
    # Using ideal gas: ρ = P*M/(R*T)
    # M = 17.4 g/mol (natural gas), R = 8314 J/(kmol·K)
    # At 50 bar: ρ = 50e5 * 17.4 / (8314 * 293) = 35.7 kg/m³
    
    fluid.gas_density = 35.7      # kg/m³ at 50 bar
    fluid.oil_density = 800.0     # Not used
    fluid.water_density = 1000.0  # Not used
    
    fluid.gas_viscosity = 0.000015   # Pa.s
    fluid.oil_viscosity = 0.001      # Not used
    fluid.water_viscosity = 0.001    # Not used
    
    # Pure gas
    fluid.gas_fraction = 1.0
    fluid.oil_fraction = 0.0
    fluid.water_fraction = 0.0
    
    print(f"Gas density: {fluid.mixture_density():.1f} kg/m³")
    print(f"Gas viscosity: {fluid.mixture_viscosity()*1000:.3f} cP")
    
    # Set pressures
    network.set_pressure(inlet, 50e5)   # 50 bar
    network.set_pressure(outlet, 48e5)  # 48 bar
    
    # Solve
    solver = ps.SteadyStateSolver(network, fluid)
    results = solver.solve()
    
    print(f"\nConverged: {results.converged}")
    if results.converged:
        flow = results.pipe_flow_rates['GAS-PIPE']
        velocity = flow / (np.pi * 0.25**2)
        
        # Convert to standard conditions
        # Assuming isothermal flow
        flow_std = flow * 50 / 1.013  # Approximate
        flow_mmscfd = flow_std * 86400 / 28316.8 * 35.315
        
        print(f"Flow rate: {flow:.4f} m³/s (actual)")
        print(f"Flow rate: {flow_mmscfd:.2f} MMSCFD (standard)")
        print(f"Velocity: {velocity:.2f} m/s")
        print(f"Pressure drop: {2:.2f} bar")


def main():
    """Run all tests"""
    print("PIPELINE SIMULATION TEST SUITE")
    print("="*60)
    
    tests = [
        ("Single Pipe", test_single_pipe),
        ("Three Node Network", test_three_node_network),
        ("Elevation Change", test_elevation_change),
        ("Gas Flow", test_gas_flow)
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, "PASSED"))
        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            results.append((name, "FAILED"))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, status in results:
        symbol = "✓" if status == "PASSED" else "✗"
        print(f"{symbol} {name}: {status}")
    
    passed = sum(1 for _, s in results if s == "PASSED")
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")


if __name__ == "__main__":
    main()