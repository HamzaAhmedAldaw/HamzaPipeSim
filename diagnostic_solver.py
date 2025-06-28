"""
Detailed diagnostic for solver issues
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

print("Solver Diagnostic - Detailed Analysis")
print("="*60)

# Test 1: Mixed BC diagnostic
print("\n1. MIXED BOUNDARY CONDITIONS TEST")
print("-"*40)

net = ps.Network()
inlet = net.add_node("IN", ps.NodeType.SOURCE)
outlet = net.add_node("OUT", ps.NodeType.SINK)
pipe = net.add_pipe("P", inlet, outlet, 100.0, 0.1)
pipe.roughness = 0.000045

# Check what BCs are set
net.set_pressure(inlet, 5e5)     # 5 bar at inlet
net.set_flow_rate(outlet, -0.01)  # 0.01 m³/s out

print(f"Network setup:")
print(f"  Nodes: {net.node_count()}")
print(f"  Pipes: {net.pipe_count()}")
print(f"  Pressure BCs: {len(net.pressure_specs)}")
print(f"  Flow BCs: {len(net.flow_specs)}")

fluid = ps.FluidProperties()
fluid.oil_density = 1000.0
fluid.oil_viscosity = 0.001
fluid.oil_fraction = 1.0

solver = ps.SteadyStateSolver(net, fluid)
solver.config.verbose = True
solver.config.max_iterations = 20
solver.config.tolerance = 1e-6

print("\nSolving mixed BC problem...")
results = solver.solve()

print(f"\nResults:")
print(f"  Converged: {results.converged}")
print(f"  Iterations: {results.iterations}")
print(f"  Residual: {results.residual:.2e}")

if results.converged:
    print(f"  Outlet pressure: {results.node_pressures['OUT']/1e5:.2f} bar")
    print(f"  Flow rate: {results.pipe_flow_rates['P']:.4f} m³/s")

# Test 2: Laminar flow with correct formula
print("\n\n2. LAMINAR FLOW DIAGNOSTIC")
print("-"*40)

net2 = ps.Network()
n1 = net2.add_node("N1", ps.NodeType.SOURCE)
n2 = net2.add_node("N2", ps.NodeType.SINK)

D = 0.01  # 10mm
L = 5.0   # 5m
pipe = net2.add_pipe("P", n1, n2, L, D)
pipe.roughness = 0.0  # Smooth pipe for laminar

# Very small pressure drop for laminar
dp = 100  # 100 Pa
net2.set_pressure(n1, 1e5 + dp)
net2.set_pressure(n2, 1e5)

# High viscosity
fluid2 = ps.FluidProperties()
fluid2.oil_density = 900.0
fluid2.oil_viscosity = 0.1  # 100 cP
fluid2.oil_fraction = 1.0

print(f"Setup:")
print(f"  Pipe: L={L}m, D={D}m")
print(f"  ΔP: {dp} Pa")
print(f"  Fluid: ρ={fluid2.mixture_density()} kg/m³, μ={fluid2.mixture_viscosity()} Pa.s")

# Analytical solution
mu = fluid2.mixture_viscosity()
Q_analytical = np.pi * D**4 * dp / (128 * mu * L)
v_analytical = Q_analytical / (np.pi * D**2 / 4)
Re_analytical = fluid2.mixture_density() * v_analytical * D / mu

print(f"\nAnalytical solution:")
print(f"  Q = π*D⁴*ΔP/(128*μ*L) = {Q_analytical:.6e} m³/s")
print(f"  v = {v_analytical:.6f} m/s")
print(f"  Re = {Re_analytical:.1f}")

solver2 = ps.SteadyStateSolver(net2, fluid2)
solver2.config.verbose = False
results2 = solver2.solve()

if results2.converged:
    Q_solver = results2.pipe_flow_rates['P']
    print(f"\nSolver result:")
    print(f"  Q = {Q_solver:.6e} m³/s")
    print(f"  Error = {abs(Q_solver - Q_analytical)/Q_analytical * 100:.1f}%")

# Test 3: Network mass balance
print("\n\n3. NETWORK MASS BALANCE TEST")
print("-"*40)

net3 = ps.Network()
s = net3.add_node("S", ps.NodeType.SOURCE)
j = net3.add_node("J", ps.NodeType.JUNCTION) 
s1 = net3.add_node("S1", ps.NodeType.SINK)
s2 = net3.add_node("S2", ps.NodeType.SINK)

p1 = net3.add_pipe("P1", s, j, 100.0, 0.15)
p2 = net3.add_pipe("P2", j, s1, 50.0, 0.1)
p3 = net3.add_pipe("P3", j, s2, 50.0, 0.1)

net3.set_pressure(s, 5e5)
net3.set_pressure(s1, 1e5)
net3.set_pressure(s2, 1e5)

solver3 = ps.SteadyStateSolver(net3, fluid)
solver3.config.verbose = False
solver3.config.tolerance = 1e-12  # Very tight tolerance
solver3.config.max_iterations = 100

results3 = solver3.solve()

if results3.converged:
    q1 = results3.pipe_flow_rates['P1']
    q2 = results3.pipe_flow_rates['P2'] 
    q3 = results3.pipe_flow_rates['P3']
    
    print(f"Flows:")
    print(f"  P1 (in): {q1:.6f} m³/s")
    print(f"  P2 (out): {q2:.6f} m³/s")
    print(f"  P3 (out): {q3:.6f} m³/s")
    print(f"  Sum out: {q2+q3:.6f} m³/s")
    print(f"  Imbalance: {abs(q1-q2-q3):.2e} m³/s")

# Test 4: Check BC handling
print("\n\n4. BOUNDARY CONDITION HANDLING")
print("-"*40)

# Test with only flow BCs (should fail)
net4 = ps.Network()
n1 = net4.add_node("N1", ps.NodeType.SOURCE)
n2 = net4.add_node("N2", ps.NodeType.SINK)
net4.add_pipe("P", n1, n2, 100.0, 0.1)

net4.set_flow_rate(n1, 0.01)   # Flow in
net4.set_flow_rate(n2, -0.01)  # Flow out

print("Setup: Only flow BCs (no pressure reference)")
print(f"  Flow BCs: {len(net4.flow_specs)}")
print(f"  Pressure BCs: {len(net4.pressure_specs)}")

solver4 = ps.SteadyStateSolver(net4, fluid)
solver4.config.verbose = False
results4 = solver4.solve()

print(f"Result: Converged = {results4.converged}")
if not results4.converged:
    print("  (This is expected - need at least one pressure BC)")

print("\n" + "="*60)