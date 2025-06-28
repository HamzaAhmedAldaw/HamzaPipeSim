#!/usr/bin/env python3
"""
Diagnostic test to check solver functionality
"""

import sys
import time
import pipeline_sim
import numpy as np

print("=== Pipeline-Sim Diagnostic Test ===\n")

# 1. Check what's available
print("1. Checking available classes and methods:")
print(f"   Network: {hasattr(pipeline_sim, 'Network')}")
print(f"   FluidProperties: {hasattr(pipeline_sim, 'FluidProperties')}")
print(f"   SteadyStateSolver: {hasattr(pipeline_sim, 'SteadyStateSolver')}")
print(f"   SolverConfig: {hasattr(pipeline_sim, 'SolverConfig')}")
print(f"   constants: {hasattr(pipeline_sim, 'constants')}")

# 2. Create a simple test network
print("\n2. Creating simple test network:")
network = pipeline_sim.Network()
print("   ✓ Network created")

# Create two nodes
n1 = network.add_node("N1", pipeline_sim.NodeType.SOURCE)
n2 = network.add_node("N2", pipeline_sim.NodeType.SINK)
print("   ✓ Nodes created")

# Create a pipe
pipe = network.add_pipe("P1", n1, n2, 1000.0, 0.3)  # 1000m, 0.3m diameter
print("   ✓ Pipe created")

# Set boundary conditions
network.set_pressure(n1, 200e5)  # 200 bar
network.set_pressure(n2, 100e5)  # 100 bar
print("   ✓ Boundary conditions set")

# 3. Create fluid properties
print("\n3. Creating fluid properties:")
fluid = pipeline_sim.FluidProperties()
fluid.oil_fraction = 1.0
fluid.water_fraction = 0.0
fluid.gas_fraction = 0.0
fluid.oil_density = 850.0
fluid.oil_viscosity = 0.001
print(f"   Density: {fluid.mixture_density()} kg/m³")
print(f"   Viscosity: {fluid.mixture_viscosity()*1000} cP")

# 4. Create and configure solver
print("\n4. Creating solver:")
solver = pipeline_sim.SteadyStateSolver(network, fluid)
print("   ✓ Solver created")

# Try to configure if possible
if hasattr(solver, 'config'):
    print("   ✓ Config available")
    config = solver.config
    config.verbose = True
    config.tolerance = 1e-6
    config.max_iterations = 10  # Low number for testing
    print(f"   Tolerance: {config.tolerance}")
    print(f"   Max iterations: {config.max_iterations}")
else:
    print("   ✗ Config NOT available")

# 5. Solve with timeout
print("\n5. Solving simple network:")
print("   Starting solve...")

start_time = time.time()
try:
    # Create a simple timeout mechanism
    import threading
    
    result = None
    exception = None
    
    def solve_thread():
        global result, exception
        try:
            result = solver.solve()
        except Exception as e:
            exception = e
    
    thread = threading.Thread(target=solve_thread)
    thread.daemon = True
    thread.start()
    
    # Wait for up to 5 seconds
    thread.join(timeout=5.0)
    
    if thread.is_alive():
        print("   ✗ Solver timeout after 5 seconds!")
        print("   The solver appears to be stuck.")
        
        # Try to get more info
        print("\n6. Network details:")
        print(f"   Nodes: {len(network.nodes())}")
        print(f"   Pipes: {len(network.pipes())}")
        
        # Check if we can access pressure/flow specs
        if hasattr(network, 'pressure_specs'):
            print(f"   Pressure BCs: {len(network.pressure_specs())}")
        if hasattr(network, 'flow_specs'):
            print(f"   Flow BCs: {len(network.flow_specs())}")
            
    elif exception:
        print(f"   ✗ Solver failed with error: {exception}")
    elif result:
        elapsed = time.time() - start_time
        print(f"   ✓ Solved in {elapsed:.3f} seconds")
        print(f"   Converged: {result.converged}")
        print(f"   Iterations: {result.iterations}")
        
        # Check results
        if result.converged:
            print("\n7. Solution results:")
            for node_id, pressure in result.node_pressures.items():
                print(f"   Node {node_id}: {pressure/1e5:.1f} bar")
            
            for pipe_id, flow in result.pipe_flow_rates.items():
                print(f"   Pipe {pipe_id}: {flow*3600:.1f} m³/h")
                
            # Calculate expected flow manually
            dp = 100e5  # 100 bar pressure drop
            L = 1000    # m
            D = 0.3     # m
            mu = 0.001  # Pa.s
            rho = 850   # kg/m³
            
            # Simplified calculation assuming laminar flow
            Q_expected = np.pi * D**4 * dp / (128 * mu * L)
            print(f"\n   Expected flow (laminar): {Q_expected*3600:.1f} m³/h")
            
except Exception as e:
    print(f"   ✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Diagnostic Complete ===")

# Additional checks
print("\n8. Additional checks:")

# Check if the module has proper version
if hasattr(pipeline_sim, '__version__'):
    print(f"   Version: {pipeline_sim.__version__}")
else:
    print("   Version: Not available")

# Check for get_version function
if hasattr(pipeline_sim, 'get_version'):
    try:
        version = pipeline_sim.get_version()
        print(f"   get_version(): {version}")
    except:
        print("   get_version(): Failed to call")

print("\nIf the solver is stuck, it likely means:")
print("1. The matrix assembly has issues (singular matrix)")
print("2. The boundary condition application is incorrect")
print("3. The solver configuration is not being applied")
print("4. There's an infinite loop in the solver")