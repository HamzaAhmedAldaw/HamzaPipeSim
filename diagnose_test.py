#!/usr/bin/env python3
"""
Diagnose why pressure boundary conditions aren't being recognized
"""

import pipeline_sim as ps

def diagnose_pressure_bc():
    print("DIAGNOSING PRESSURE BOUNDARY CONDITIONS")
    print("="*70)
    
    # Create simple network
    network = ps.Network()
    
    # Create two nodes
    inlet = network.add_node("INLET", ps.NodeType.SOURCE)
    outlet = network.add_node("OUTLET", ps.NodeType.SINK)
    
    # Create pipe
    pipe = network.add_pipe("PIPE", inlet, outlet, 1000.0, 0.3)
    
    print(f"\nNetwork created: {network.node_count()} nodes, {network.pipe_count()} pipes")
    
    # Check initial state
    print("\nInitial node states:")
    print(f"  Inlet pressure: {inlet.pressure/1e5:.2f} bar")
    print(f"  Inlet has_pressure_bc: {inlet.has_pressure_bc()}")
    print(f"  Outlet pressure: {outlet.pressure/1e5:.2f} bar")
    print(f"  Outlet has_pressure_bc: {outlet.has_pressure_bc()}")
    
    # Check network pressure specs
    print(f"\nNetwork pressure_specs: {dict(network.pressure_specs())}")
    
    # Try setting pressure using network method
    print("\nSetting pressures via network.set_pressure()...")
    network.set_pressure(inlet, 10e5)   # 10 bar
    network.set_pressure(outlet, 9e5)   # 9 bar
    
    # Check after network.set_pressure
    print("\nAfter network.set_pressure():")
    print(f"  Inlet pressure: {inlet.pressure/1e5:.2f} bar")
    print(f"  Inlet has_pressure_bc: {inlet.has_pressure_bc()}")
    print(f"  Outlet pressure: {outlet.pressure/1e5:.2f} bar")
    print(f"  Outlet has_pressure_bc: {outlet.has_pressure_bc()}")
    print(f"  Network pressure_specs: {dict(network.pressure_specs())}")
    
    # Try setting pressure BC directly on node
    print("\nTrying to set pressure BC directly on nodes...")
    inlet.set_pressure_bc(10e5)
    outlet.set_pressure_bc(9e5)
    
    print("\nAfter node.set_pressure_bc():")
    print(f"  Inlet has_pressure_bc: {inlet.has_pressure_bc()}")
    print(f"  Inlet pressure_bc: {inlet.pressure_bc()/1e5:.2f} bar")
    print(f"  Outlet has_pressure_bc: {outlet.has_pressure_bc()}")
    print(f"  Outlet pressure_bc: {outlet.pressure_bc()/1e5:.2f} bar")
    
    # Create fluid
    fluid = ps.FluidProperties()
    fluid.oil_density = 850.0
    fluid.oil_viscosity = 0.002
    fluid.oil_fraction = 1.0
    fluid.gas_fraction = 0.0
    fluid.water_fraction = 0.0
    
    # Try to solve
    print("\nCreating solver and attempting to solve...")
    solver = ps.SteadyStateSolver(network, fluid)
    solver.config.verbose = False
    
    results = solver.solve()
    
    print(f"\nSolver results:")
    print(f"  Converged: {results.converged}")
    print(f"  Iterations: {results.iterations}")
    print(f"  Reason: {results.convergence_reason}")
    
    return results


def test_direct_bc_setting():
    """Test setting BCs directly on nodes"""
    print("\n" + "="*70)
    print("TEST: SETTING BCs DIRECTLY ON NODES")
    print("="*70)
    
    network = ps.Network()
    
    # Create simple 3-node network
    source = network.add_node("SOURCE", ps.NodeType.SOURCE)
    junction = network.add_node("JUNCTION", ps.NodeType.JUNCTION)
    sink = network.add_node("SINK", ps.NodeType.SINK)
    
    # Pipes
    pipe1 = network.add_pipe("PIPE1", source, junction, 1000, 0.3)
    pipe2 = network.add_pipe("PIPE2", junction, sink, 1000, 0.3)
    
    # Set BCs directly
    print("\nSetting boundary conditions directly on nodes...")
    source.set_pressure_bc(10e5)  # 10 bar
    sink.set_pressure_bc(5e5)      # 5 bar
    
    print(f"Source has BC: {source.has_pressure_bc()}, value: {source.pressure_bc()/1e5:.1f} bar")
    print(f"Sink has BC: {sink.has_pressure_bc()}, value: {sink.pressure_bc()/1e5:.1f} bar")
    print(f"Junction has BC: {junction.has_pressure_bc()}")
    
    # Fluid
    fluid = ps.FluidProperties()
    fluid.oil_density = 850.0
    fluid.oil_viscosity = 0.002
    fluid.oil_fraction = 1.0
    
    # Solve
    solver = ps.SteadyStateSolver(network, fluid)
    solver.config.verbose = True
    solver.config.max_iterations = 50
    
    print("\nSolving with direct BC setting...")
    results = solver.solve()
    
    print(f"\nResults:")
    print(f"  Converged: {results.converged}")
    print(f"  Iterations: {results.iterations}")
    
    if results.converged:
        print(f"  Junction pressure: {results.node_pressures.get('JUNCTION', 0)/1e5:.2f} bar")
        print(f"  Flow rate: {results.pipe_flow_rates.get('PIPE1', 0):.4f} m³/s")
    
    return results


def main():
    print("PRESSURE BOUNDARY CONDITION DIAGNOSTIC")
    print("Version:", ps.__version__)
    print("")
    
    # Test 1: Diagnose BC setting
    result1 = diagnose_pressure_bc()
    
    # Test 2: Try direct BC setting
    result2 = test_direct_bc_setting()
    
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    
    if result2 and result2.iterations > 0:
        print("✓ Solver works when BCs are set directly on nodes")
        print("✗ Issue: network.set_pressure() is not properly setting BCs")
        print("\nWORKAROUND: Use node.set_pressure_bc() instead of network.set_pressure()")
    else:
        print("✗ Solver still not working even with direct BC setting")
        print("  There may be additional issues beyond BC setting")


if __name__ == "__main__":
    main()