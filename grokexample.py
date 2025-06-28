#!/usr/bin/env python3
"""
Pipeline Simulation - Testing Real-World Data vs Implementation Issues
"""

import os
import sys
import logging
from pathlib import Path
import math
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from loader
from pipeline_sim_loader import pipeline_sim as ps

def test_with_realistic_oil_pipeline():
    """Test with real-world oil pipeline parameters"""
    
    logger.info("TESTING WITH REALISTIC OIL PIPELINE DATA")
    logger.info("="*60)
    
    try:
        # Real pipeline example: Trans-Alaska Pipeline System (TAPS) segment
        network = ps.Network()
        
        # Pump station to next station
        pump_station = network.add_node("PS01", ps.NodeType.SOURCE)
        next_station = network.add_node("PS02", ps.NodeType.SINK)
        
        # Real elevations (meters above sea level)
        pump_station.elevation = 150.0  # Typical elevation
        next_station.elevation = 200.0  # 50m elevation gain
        
        # TAPS specs: 48 inch (1.22m) diameter, ~100 km between stations
        pipe = network.add_pipe(
            "TAPS_Segment",
            pump_station,
            next_station,
            100000.0,  # 100 km
            1.22       # 48 inches
        )
        
        # Steel pipe roughness
        pipe.roughness = 0.000046  # Commercial steel
        pipe.inclination = math.atan2(50.0, 100000.0)  # Small incline
        
        # Prudhoe Bay crude oil properties at 60°F
        fluid = ps.FluidProperties()
        fluid.oil_density = 870.0      # kg/m³ (typical Alaska crude)
        fluid.oil_viscosity = 0.015    # Pa·s at operating temp
        fluid.water_density = 1000.0   
        fluid.gas_density = 1.2        
        fluid.water_viscosity = 0.001  
        fluid.gas_viscosity = 1.8e-5   
        
        # Single phase oil
        fluid.oil_fraction = 1.0
        fluid.gas_fraction = 0.0
        fluid.water_fraction = 0.0
        
        # Typical TAPS operating conditions
        # Discharge pressure: ~1000 psi (69 bar)
        # Flow rate: ~1.8 million barrels/day = ~3.3 m³/s
        network.set_pressure(pump_station, 69e5)  # 69 bar
        network.set_flow_rate(next_station, 3.3)   # 3.3 m³/s
        
        logger.info("Pipeline specifications:")
        logger.info(f"  Length: {pipe.length/1000:.1f} km")
        logger.info(f"  Diameter: {pipe.diameter:.2f} m")
        logger.info(f"  Flow rate: {3.3:.1f} m³/s")
        logger.info(f"  Inlet pressure: 69 bar")
        
        # Calculate expected values manually
        area = math.pi * pipe.diameter**2 / 4
        velocity = 3.3 / area
        Re = fluid.oil_density * velocity * pipe.diameter / fluid.oil_viscosity
        
        logger.info(f"\nExpected flow conditions:")
        logger.info(f"  Velocity: {velocity:.2f} m/s")
        logger.info(f"  Reynolds number: {Re:.0e}")
        logger.info(f"  Flow regime: {'Turbulent' if Re > 4000 else 'Laminar'}")
        
        # Expected pressure drop (Darcy-Weisbach)
        if Re > 4000:  # Turbulent
            # Colebrook-White approximation
            f = 0.02  # Typical for large pipes
        else:
            f = 64 / Re
            
        dp_friction = f * (pipe.length / pipe.diameter) * (0.5 * fluid.oil_density * velocity**2)
        dp_elevation = fluid.oil_density * 9.81 * 50  # Elevation change
        dp_total = dp_friction + dp_elevation
        
        logger.info(f"\nExpected pressure drops:")
        logger.info(f"  Friction: {dp_friction/1e5:.1f} bar")
        logger.info(f"  Elevation: {dp_elevation/1e5:.1f} bar")
        logger.info(f"  Total: {dp_total/1e5:.1f} bar")
        
        # Try to solve
        solver = ps.SteadyStateSolver(network, fluid)
        config = solver.config()
        config.tolerance = 1e-4  # Looser tolerance for large system
        config.max_iterations = 200
        config.verbose = True
        
        results = solver.solve()
        logger.info(f"\nSolver result: Converged = {results.converged}")
        
        if results.converged:
            logger.info("✓ SUCCESS with realistic data!")
        else:
            logger.warning("✗ Failed even with realistic data")
            
    except Exception as e:
        logger.error(f"Realistic test failed: {str(e)}")
        import traceback
        traceback.print_exc()

def test_implementation_issues():
    """Test for specific C++ implementation issues"""
    
    logger.info("\n\nTESTING FOR C++ IMPLEMENTATION ISSUES")
    logger.info("="*60)
    
    try:
        # Issue 1: Matrix size mismatch
        logger.info("\n1. Testing matrix dimensions...")
        network = ps.Network()
        n1 = network.add_node("n1", ps.NodeType.SOURCE)
        n2 = network.add_node("n2", ps.NodeType.SINK)
        pipe = network.add_pipe("p1", n1, n2, 100.0, 0.3)
        
        # The solver needs N equations for N unknowns
        # With 2 nodes and 1 pipe, we have:
        # - 2 pressure unknowns (but 1 is fixed by BC)
        # - 1 flow unknown (but it's fixed by BC)
        # So we should have 0 unknowns... this might be the issue!
        
        logger.info(f"  Nodes: {network.node_count()}")
        logger.info(f"  Pipes: {network.pipe_count()}")
        logger.info(f"  Pressure BCs: {len(network.pressure_specs())}")
        logger.info(f"  Flow BCs: {len(network.flow_specs())}")
        
        # Try with more nodes to have actual unknowns
        logger.info("\n2. Testing with junction node (3 nodes, 2 pipes)...")
        network2 = ps.Network()
        n1 = network2.add_node("n1", ps.NodeType.SOURCE)
        n2 = network2.add_node("n2", ps.NodeType.JUNCTION)  # No BC here
        n3 = network2.add_node("n3", ps.NodeType.SINK)
        
        p1 = network2.add_pipe("p1", n1, n2, 100.0, 0.3)
        p2 = network2.add_pipe("p2", n2, n3, 100.0, 0.3)
        
        # Now we have 3 nodes, 2 pipes
        # BCs: 1 pressure (n1), 1 flow (n3)
        # Unknowns: n2 pressure, p1 flow
        network2.set_pressure(n1, 2e5)
        network2.set_flow_rate(n3, 0.1)
        
        fluid = ps.FluidProperties()
        fluid.oil_fraction = 1.0
        fluid.gas_fraction = 0.0
        fluid.water_fraction = 0.0
        
        solver2 = ps.SteadyStateSolver(network2, fluid)
        results2 = solver2.solve()
        logger.info(f"  With junction: Converged = {results2.converged}")
        
        # Issue 3: Maybe the C++ code expects initialized values
        logger.info("\n3. Testing with pre-initialized pressures...")
        network3 = ps.Network()
        n1 = network3.add_node("n1", ps.NodeType.SOURCE)
        n2 = network3.add_node("n2", ps.NodeType.SINK)
        pipe = network3.add_pipe("p1", n1, n2, 100.0, 0.3)
        
        # Set initial guesses before BCs
        n1.pressure = 2e5
        n2.pressure = 1.8e5  # Slightly lower
        
        # Then set BCs
        network3.set_pressure(n1, 2e5)
        network3.set_flow_rate(n2, 0.1)
        
        solver3 = ps.SteadyStateSolver(network3, fluid)
        results3 = solver3.solve()
        logger.info(f"  With initialization: Converged = {results3.converged}")
        
    except Exception as e:
        logger.error(f"Implementation test failed: {str(e)}")
        import traceback
        traceback.print_exc()

def analyze_cpp_error():
    """Analyze the specific C++ error"""
    
    logger.info("\n\nANALYZING C++ MATRIX DECOMPOSITION ERROR")
    logger.info("="*60)
    
    logger.info("""
The "Matrix decomposition failed!" error suggests:

1. **Singular Matrix**: The system matrix is singular (non-invertible)
   - This happens when the system is over/under-constrained
   - With 2 nodes and both having BCs, there might be no degrees of freedom

2. **Implementation Bug**: The C++ solver might have issues with:
   - Boundary condition application
   - Matrix assembly
   - Node/equation indexing

3. **Missing Initialization**: The solver might expect:
   - Initial pressure/flow guesses
   - Proper node type handling
   - Specific boundary condition combinations

4. **Numerical Issues**: 
   - Very small/large values causing conditioning problems
   - Unit inconsistencies (Pa vs bar, etc.)
""")
    
    # Test the specific theory about degrees of freedom
    logger.info("\nTesting degrees of freedom theory...")
    
    try:
        # Case 1: Over-constrained (2 nodes, 2 BCs)
        network = ps.Network()
        n1 = network.add_node("n1", ps.NodeType.SOURCE)
        n2 = network.add_node("n2", ps.NodeType.SINK)
        network.add_pipe("p1", n1, n2, 100.0, 0.3)
        
        # This might be over-constrained!
        network.set_pressure(n1, 2e5)
        network.set_flow_rate(n2, 0.1)
        
        logger.info("Case 1: 2 nodes, 1 pressure BC, 1 flow BC")
        logger.info("  Degrees of freedom: Possibly 0 (over-constrained)")
        
        # Case 2: Try with only one BC
        network2 = ps.Network()
        n1 = network2.add_node("n1", ps.NodeType.SOURCE)
        n2 = network2.add_node("n2", ps.NodeType.SINK)
        network2.add_pipe("p1", n1, n2, 100.0, 0.3)
        
        # Only set inlet pressure, let flow be computed
        network2.set_pressure(n1, 2e5)
        network2.set_pressure(n2, 1e5)  # Two pressures instead
        
        fluid = ps.FluidProperties()
        fluid.oil_fraction = 1.0
        fluid.gas_fraction = 0.0
        fluid.water_fraction = 0.0
        
        solver = ps.SteadyStateSolver(network2, fluid)
        results = solver.solve()
        logger.info(f"\nTwo pressure BCs: Converged = {results.converged}")
        
    except Exception as e:
        logger.error(f"DOF analysis failed: {str(e)}")

def suggest_fixes():
    """Suggest fixes for the C++ implementation"""
    
    logger.info("\n\nSUGGESTED C++ FIXES")
    logger.info("="*60)
    
    logger.info("""
Based on the analysis, the C++ solver likely needs fixes:

1. **Matrix Assembly Fix**:
   - Check degrees of freedom calculation
   - Ensure proper equation indexing
   - Handle 2-node networks specially

2. **Boundary Condition Application**:
   - Apply BC values to node pressures before matrix assembly
   - Ensure BC nodes are properly marked in the matrix

3. **Add Validation**:
   - Check for minimum network size (maybe need 3+ nodes?)
   - Validate BC combinations
   - Better error messages than "Matrix decomposition failed!"

4. **Numerical Improvements**:
   - Add matrix conditioning checks
   - Use pivoting in decomposition
   - Scale equations for better conditioning

The issue appears to be in the C++ core, not the data.
""")

if __name__ == "__main__":
    try:
        # Test with realistic data
        test_with_realistic_oil_pipeline()
        
        # Test for implementation issues
        test_implementation_issues()
        
        # Analyze the error
        analyze_cpp_error()
        
        # Suggest fixes
        suggest_fixes()
        
        logger.info("\n" + "="*60)
        logger.info("CONCLUSION: The issue is likely in the C++ solver implementation,")
        logger.info("not in the data. The solver fails before iterations begin,")
        logger.info("suggesting a matrix assembly or degrees of freedom problem.")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Program failed: {str(e)}")
        sys.exit(1)