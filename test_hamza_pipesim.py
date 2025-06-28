"""
Comprehensive Pipeline Test Suite
Tests various configurations to verify solver robustness
"""

import sys
import importlib.util
from pathlib import Path
import numpy as np

# Load the module
pyd_path = Path(r"C:\Users\KIMO STORE\HamzaPipeSim\build\lib.win-amd64-cpython-313\pipeline_sim.cp313-win_amd64.pyd")
spec = importlib.util.spec_from_file_location("pipeline_sim", str(pyd_path))
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)

print("Pipeline-Sim Comprehensive Test Suite")
print("="*60)

def run_test(name, test_func):
    """Run a test and report results"""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print("="*60)
    try:
        result = test_func()
        if result:
            print(f"✅ {name} PASSED")
        else:
            print(f"❌ {name} FAILED")
        return result
    except Exception as e:
        print(f"❌ {name} FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_horizontal_pipe():
    """Test 1: Simple horizontal pipe"""
    net = ps.Network()
    n1 = net.add_node("IN", ps.NodeType.SOURCE)
    n2 = net.add_node("OUT", ps.NodeType.SINK)
    pipe = net.add_pipe("P1", n1, n2, 100.0, 0.1)
    pipe.roughness = 0.000045
    
    net.set_pressure(n1, 2e5)  # 2 bar
    net.set_pressure(n2, 1e5)  # 1 bar
    
    fluid = ps.FluidProperties()
    fluid.oil_density = 1000.0
    fluid.oil_viscosity = 0.001
    fluid.oil_fraction = 1.0
    
    solver = ps.SteadyStateSolver(net, fluid)
    solver.config.verbose = False
    results = solver.solve()
    
    if results.converged:
        flow = results.pipe_flow_rates['P1']
        velocity = flow / (np.pi * 0.05**2)
        Re = 1000 * velocity * 0.1 / 0.001
        
        print(f"  Flow: {flow:.4f} m³/s ({flow*3600:.1f} m³/h)")
        print(f"  Velocity: {velocity:.2f} m/s")
        print(f"  Reynolds: {Re:.0f}")
        
        # Validate results
        return flow > 0 and velocity > 0 and Re > 2300  # Turbulent flow expected
    return False

def test_vertical_pipe():
    """Test 2: Vertical pipe with elevation change"""
    net = ps.Network()
    bottom = net.add_node("BOTTOM", ps.NodeType.SOURCE)
    top = net.add_node("TOP", ps.NodeType.SINK)
    
    # 10m vertical rise
    bottom.elevation = 0.0
    top.elevation = 10.0
    
    pipe = net.add_pipe("RISER", bottom, top, 10.0, 0.05)
    pipe.roughness = 0.000045
    
    # Need to overcome hydrostatic head (~1 bar for 10m water)
    net.set_pressure(bottom, 3e5)  # 3 bar
    net.set_pressure(top, 1e5)    # 1 bar
    
    fluid = ps.FluidProperties()
    fluid.oil_density = 1000.0
    fluid.oil_viscosity = 0.001
    fluid.oil_fraction = 1.0
    
    solver = ps.SteadyStateSolver(net, fluid)
    solver.config.verbose = False
    results = solver.solve()
    
    if results.converged:
        flow = results.pipe_flow_rates['RISER']
        dp_total = results.pipe_pressure_drops['RISER']
        dp_hydrostatic = 1000 * 9.81 * 10  # ρgh
        dp_friction = dp_total - dp_hydrostatic
        
        print(f"  Flow: {flow:.4f} m³/s")
        print(f"  Total ΔP: {dp_total/1e5:.2f} bar")
        print(f"  Hydrostatic ΔP: {dp_hydrostatic/1e5:.2f} bar")
        print(f"  Friction ΔP: {dp_friction/1e5:.2f} bar")
        
        return flow > 0 and dp_friction > 0
    return False

def test_network():
    """Test 3: Branching network"""
    net = ps.Network()
    
    # Create nodes
    source = net.add_node("SOURCE", ps.NodeType.SOURCE)
    junction = net.add_node("JUNC", ps.NodeType.JUNCTION)
    sink1 = net.add_node("SINK1", ps.NodeType.SINK)
    sink2 = net.add_node("SINK2", ps.NodeType.SINK)
    
    # Create pipes
    supply = net.add_pipe("SUPPLY", source, junction, 100.0, 0.15)
    branch1 = net.add_pipe("BR1", junction, sink1, 50.0, 0.1)
    branch2 = net.add_pipe("BR2", junction, sink2, 50.0, 0.1)
    
    for p in [supply, branch1, branch2]:
        p.roughness = 0.000045
    
    # Set boundary conditions
    net.set_pressure(source, 5e5)   # 5 bar
    net.set_pressure(sink1, 1e5)    # 1 bar
    net.set_pressure(sink2, 1e5)    # 1 bar
    
    fluid = ps.FluidProperties()
    fluid.oil_density = 850.0    # Light oil
    fluid.oil_viscosity = 0.005  # 5 cP
    fluid.oil_fraction = 1.0
    
    solver = ps.SteadyStateSolver(net, fluid)
    solver.config.verbose = False
    results = solver.solve()
    
    if results.converged:
        q_supply = results.pipe_flow_rates['SUPPLY']
        q_br1 = results.pipe_flow_rates['BR1']
        q_br2 = results.pipe_flow_rates['BR2']
        p_junc = results.node_pressures['JUNC']
        
        print(f"  Supply flow: {q_supply:.4f} m³/s")
        print(f"  Branch 1 flow: {q_br1:.4f} m³/s")
        print(f"  Branch 2 flow: {q_br2:.4f} m³/s")
        print(f"  Junction pressure: {p_junc/1e5:.2f} bar")
        
        # Check mass balance
        imbalance = abs(q_supply - q_br1 - q_br2)
        print(f"  Mass balance error: {imbalance:.2e} m³/s")
        
        return imbalance < 1e-10
    return False

def test_mixed_bc():
    """Test 4: Mixed boundary conditions (pressure + flow)"""
    net = ps.Network()
    inlet = net.add_node("INLET", ps.NodeType.SOURCE)
    outlet = net.add_node("OUTLET", ps.NodeType.SINK)
    pipe = net.add_pipe("PIPE", inlet, outlet, 200.0, 0.2)
    pipe.roughness = 0.000045
    
    # Pressure at inlet, flow at outlet
    net.set_pressure(inlet, 10e5)      # 10 bar
    net.set_flow_rate(outlet, -0.1)    # 0.1 m³/s out
    
    fluid = ps.FluidProperties()
    fluid.oil_density = 900.0
    fluid.oil_viscosity = 0.01  # 10 cP
    fluid.oil_fraction = 1.0
    
    solver = ps.SteadyStateSolver(net, fluid)
    solver.config.verbose = False
    solver.config.tolerance = 1e-8
    results = solver.solve()
    
    if results.converged:
        flow = results.pipe_flow_rates['PIPE']
        p_out = results.node_pressures['OUTLET']
        dp = results.pipe_pressure_drops['PIPE']
        
        print(f"  Flow: {flow:.4f} m³/s (specified: 0.1)")
        print(f"  Outlet pressure: {p_out/1e5:.2f} bar")
        print(f"  Pressure drop: {dp/1e5:.2f} bar")
        
        return abs(flow - 0.1) < 1e-6
    return False

def test_laminar_flow():
    """Test 5: Laminar flow validation"""
    net = ps.Network()
    n1 = net.add_node("N1", ps.NodeType.SOURCE)
    n2 = net.add_node("N2", ps.NodeType.SINK)
    
    # Small diameter, high viscosity for laminar flow
    D = 0.01  # 10mm
    L = 5.0   # 5m
    pipe = net.add_pipe("P", n1, n2, L, D)
    
    dp = 1000  # 1000 Pa = 0.01 bar
    net.set_pressure(n1, 1e5 + dp)
    net.set_pressure(n2, 1e5)
    
    # High viscosity for laminar flow
    fluid = ps.FluidProperties()
    fluid.oil_density = 900.0
    fluid.oil_viscosity = 0.1  # 100 cP
    fluid.oil_fraction = 1.0
    
    solver = ps.SteadyStateSolver(net, fluid)
    solver.config.verbose = False
    results = solver.solve()
    
    if results.converged:
        Q_actual = results.pipe_flow_rates['P']
        
        # Hagen-Poiseuille: Q = π*D⁴*ΔP / (128*μ*L)
        mu = fluid.mixture_viscosity()
        Q_theory = np.pi * D**4 * dp / (128 * mu * L)
        
        error = abs(Q_actual - Q_theory) / Q_theory * 100
        
        v = Q_actual / (np.pi * D**2 / 4)
        Re = fluid.mixture_density() * v * D / mu
        
        print(f"  Flow actual: {Q_actual:.6f} m³/s")
        print(f"  Flow theory: {Q_theory:.6f} m³/s")
        print(f"  Error: {error:.2f}%")
        print(f"  Reynolds: {Re:.1f} (should be < 2300)")
        
        return error < 5.0 and Re < 2300
    return False

# Run all tests
tests = [
    ("Horizontal Pipe", test_horizontal_pipe),
    ("Vertical Pipe", test_vertical_pipe),
    ("Branching Network", test_network),
    ("Mixed Boundary Conditions", test_mixed_bc),
    ("Laminar Flow Validation", test_laminar_flow),
]

results = []
for name, test_func in tests:
    passed = run_test(name, test_func)
    results.append((name, passed))

# Summary
print(f"\n{'='*60}")
print("TEST SUMMARY")
print("="*60)

passed_count = 0
for name, passed in results:
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"{name:.<40} {status}")
    if passed:
        passed_count += 1

print(f"\nTotal: {passed_count}/{len(tests)} tests passed")

if passed_count == len(tests):
    print("\n🎉 ALL TESTS PASSED! Pipeline-Sim is working perfectly!")
else:
    print(f"\n⚠️  {len(tests) - passed_count} tests failed. Check the implementation.")