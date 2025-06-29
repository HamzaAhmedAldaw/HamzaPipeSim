#!/usr/bin/env python3
"""
Minimal test to verify the solver works at all
"""

import pipeline_sim as ps
import time

def test_minimal():
    print("MINIMAL SOLVER TEST")
    print("="*50)
    
    # Create the simplest possible network
    network = ps.Network()
    
    inlet = network.add_node("IN", ps.NodeType.SOURCE)
    outlet = network.add_node("OUT", ps.NodeType.SINK)
    
    # Short pipe with large diameter (easy to solve)
    pipe = network.add_pipe("PIPE", inlet, outlet, 100.0, 0.5)  # 100m, 0.5m diameter
    pipe.set_roughness(0.00005)
    
    # Small pressure difference
    inlet.set_pressure_bc(2e5)    # 2 bar
    outlet.set_pressure_bc(1.9e5)  # 1.9 bar
    
    # Simple fluid
    fluid = ps.FluidProperties()
    fluid.oil_fraction = 1.0
    fluid.water_fraction = 0.0
    fluid.gas_fraction = 0.0
    fluid.oil_density = 1000.0
    fluid.oil_viscosity = 0.001
    
    print("\nSetup:")
    print(f"  Pipe: {pipe.length}m long, {pipe.diameter}m diameter")
    print(f"  Pressure drop: 0.1 bar")
    print(f"  Fluid: water-like")
    
    # Create solver with minimal settings
    solver = ps.SteadyStateSolver(network, fluid)
    solver.config.verbose = True
    solver.config.max_iterations = 10  # Very low limit
    solver.config.tolerance = 1e-4     # Relaxed tolerance
    
    print("\nSolving...")
    start_time = time.time()
    
    try:
        results = solver.solve()
        elapsed = time.time() - start_time
        
        print(f"\nSolution time: {elapsed:.3f} seconds")
        
        if results.converged:
            print("✓ CONVERGED!")
            print(f"  Iterations: {results.iterations}")
            print(f"  Flow rate: {results.pipe_flow_rates['PIPE']:.4f} m³/s")
            print(f"  Velocity: {results.pipe_velocities['PIPE']:.2f} m/s")
            return True
        else:
            print("✗ FAILED TO CONVERGE")
            print(f"  Reason: {results.convergence_reason}")
            print(f"  Iterations: {results.iterations}")
            print(f"  Residual: {results.residual}")
            return False
            
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ ERROR after {elapsed:.3f} seconds: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_increasing_difficulty():
    """Test with increasing difficulty"""
    print("\n\nTEST WITH INCREASING DIFFICULTY")
    print("="*50)
    
    test_cases = [
        # (length, diameter, dp_bar, description)
        (100, 0.5, 0.1, "Easy - short, wide pipe"),
        (1000, 0.3, 0.5, "Medium - longer, narrower"),
        (304.8, 0.1541, 1.0, "Crane-like dimensions"),
        (304.8, 0.1541, 3.0, "Higher pressure drop"),
    ]
    
    for length, diameter, dp_bar, description in test_cases:
        print(f"\nTest: {description}")
        print(f"  L={length}m, D={diameter}m, ΔP={dp_bar} bar")
        
        network = ps.Network()
        inlet = network.add_node("IN", ps.NodeType.SOURCE)
        outlet = network.add_node("OUT", ps.NodeType.SINK)
        pipe = network.add_pipe("PIPE", inlet, outlet, length, diameter)
        pipe.set_roughness(0.000045)
        
        inlet.set_pressure_bc(10e5)
        outlet.set_pressure_bc((10 - dp_bar) * 1e5)
        
        fluid = ps.FluidProperties()
        fluid.oil_fraction = 1.0
        fluid.water_fraction = 0.0
        fluid.gas_fraction = 0.0
        fluid.oil_density = 1000.0
        fluid.oil_viscosity = 0.001
        
        solver = ps.SteadyStateSolver(network, fluid)
        solver.config.verbose = False
        solver.config.max_iterations = 50
        solver.config.tolerance = 1e-5
        
        start_time = time.time()
        
        try:
            results = solver.solve()
            elapsed = time.time() - start_time
            
            if results.converged:
                Q = results.pipe_flow_rates['PIPE']
                v = results.pipe_velocities['PIPE']
                Re = results.pipe_reynolds_numbers['PIPE']
                print(f"  ✓ Converged in {results.iterations} iterations ({elapsed:.3f}s)")
                print(f"    Q = {Q:.4f} m³/s, v = {v:.2f} m/s, Re = {Re:.0f}")
            else:
                print(f"  ✗ Failed after {results.iterations} iterations ({elapsed:.3f}s)")
                print(f"    Reason: {results.convergence_reason}")
                
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  ✗ Error after {elapsed:.3f}s: {e}")


def test_crane_direct():
    """Test Crane case directly without binary search"""
    print("\n\nCRANE CASE - DIRECT TEST")
    print("="*50)
    
    network = ps.Network()
    inlet = network.add_node("IN", ps.NodeType.SOURCE)
    outlet = network.add_node("OUT", ps.NodeType.SINK)
    
    # Crane dimensions
    pipe = network.add_pipe("PIPE", inlet, outlet, 304.8, 0.1541)
    pipe.set_roughness(0.000045)
    
    # Try with 44 psi pressure drop (expected value)
    dp_pa = 44 * 6894.76  # psi to Pa
    inlet.set_pressure_bc(10e5)
    outlet.set_pressure_bc(10e5 - dp_pa)
    
    fluid = ps.FluidProperties()
    fluid.oil_fraction = 1.0
    fluid.water_fraction = 0.0
    fluid.gas_fraction = 0.0
    fluid.oil_density = 999.0
    fluid.oil_viscosity = 0.00112
    
    print(f"\nDirect test with 44 psi pressure drop:")
    print(f"  Inlet: {10} bar")
    print(f"  Outlet: {(10e5 - dp_pa)/1e5:.2f} bar")
    
    solver = ps.SteadyStateSolver(network, fluid)
    solver.config.verbose = True
    solver.config.max_iterations = 30
    solver.config.tolerance = 1e-5
    solver.config.use_line_search = True
    solver.config.relaxation_factor = 0.8
    
    print("\nSolving...")
    
    try:
        results = solver.solve()
        
        if results.converged:
            Q = results.pipe_flow_rates['PIPE']
            Q_gpm = Q * 15850.3
            print(f"\n✓ Converged!")
            print(f"  Flow rate: {Q:.4f} m³/s ({Q_gpm:.0f} gpm)")
            print(f"  Target was: 1000 gpm")
            print(f"  Difference: {abs(Q_gpm - 1000):.0f} gpm")
        else:
            print(f"\n✗ Failed: {results.convergence_reason}")
            
    except Exception as e:
        print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    print("Pipeline-Sim Solver Diagnostic")
    print(f"Version: {ps.__version__}")
    print("")
    
    # Run tests
    success = test_minimal()
    
    if success:
        test_increasing_difficulty()
        test_crane_direct()
    else:
        print("\n⚠ Basic test failed - check solver configuration")
    
    print("\nDiagnostic complete")