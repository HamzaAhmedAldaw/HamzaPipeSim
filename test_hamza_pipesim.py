#!/usr/bin/env python3
"""
Test script to verify HamzaPipeSim installation
"""

import sys

def test_import():
    """Test basic import"""
    print("Testing HamzaPipeSim import...")
    try:
        import pipeline_sim
        print("✓ Successfully imported pipeline_sim")
        return True
    except ImportError as e:
        print(f"✗ Failed to import pipeline_sim: {e}")
        return False

def test_basic_functionality():
    """Test basic network creation"""
    print("\nTesting basic functionality...")
    try:
        from pipeline_sim import Network, NodeType, FluidProperties, SteadyStateSolver
        
        # Create a simple network
        network = Network()
        print("✓ Created network")
        
        # Add nodes
        node1 = network.add_node("N1", NodeType.SOURCE)
        node2 = network.add_node("N2", NodeType.JUNCTION)
        node3 = network.add_node("N3", NodeType.SINK)
        print("✓ Added 3 nodes")
        
        # Add pipes
        pipe1 = network.add_pipe("P1", node1, node2, 1000.0, 0.3)
        pipe2 = network.add_pipe("P2", node2, node3, 1500.0, 0.3)
        print("✓ Added 2 pipes")
        
        # Set boundary conditions
        network.set_pressure(node1, 500000.0)  # 5 bar
        network.set_flow_rate(node3, -0.1)     # 0.1 m³/s production
        print("✓ Set boundary conditions")
        
        # Create fluid properties
        fluid = FluidProperties()
        fluid.oil_density = 850.0
        fluid.oil_viscosity = 0.001
        print("✓ Created fluid properties")
        
        # Create and run solver
        solver = SteadyStateSolver(network, fluid)
        solver.config().tolerance = 1e-6
        solver.config().max_iterations = 100
        print("✓ Created solver")
        
        results = solver.solve()
        print("✓ Solver completed")
        
        if results.converged:
            print(f"✓ Solution converged in {results.iterations} iterations")
            print(f"  Residual: {results.residual:.2e}")
            
            # Print some results
            print("\nResults:")
            for node_id, pressure in results.node_pressures.items():
                print(f"  Node {node_id}: {pressure/1e5:.2f} bar")
            
            for pipe_id, flow in results.pipe_flow_rates.items():
                print(f"  Pipe {pipe_id}: {flow:.3f} m³/s")
        else:
            print("✗ Solution did not converge")
            
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ml_features():
    """Test ML integration features"""
    print("\nTesting ML features...")
    try:
        from pipeline_sim.ml import FeatureExtractor, FlowPatternPredictor
        
        # Test feature extractor
        extractor = FeatureExtractor()
        feature_names = extractor.get_feature_names()
        print(f"✓ Feature extractor has {len(feature_names)} features")
        
        # Test flow pattern predictor
        predictor = FlowPatternPredictor()
        print("✓ Created flow pattern predictor")
        
        return True
        
    except Exception as e:
        print(f"✗ ML features error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("HamzaPipeSim Installation Test")
    print("=" * 60)
    
    all_passed = True
    
    # Test 1: Import
    if not test_import():
        all_passed = False
        print("\nImport failed. Make sure to run:")
        print("  python setup_complete.py install")
        return 1
    
    # Test 2: Basic functionality
    if not test_basic_functionality():
        all_passed = False
    
    # Test 3: ML features
    if not test_ml_features():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed!")
        print("HamzaPipeSim is installed and working correctly.")
        return 0
    else:
        print("✗ Some tests failed.")
        print("Please check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())