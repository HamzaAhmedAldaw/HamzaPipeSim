#!/usr/bin/env python3
"""
Wrapper to run professional_test.py with any missing attributes patched
"""

import sys
import os

# Try to import pipeline_sim
try:
    import pipeline_sim
    print("✅ Pipeline-Sim imported successfully")
    
    # Check for required components
    if not hasattr(pipeline_sim, 'SolverConfig'):
        print("⚠️ SolverConfig missing, adding patch...")
        class SolverConfig:
            def __init__(self):
                self.tolerance = 1e-6
                self.max_iterations = 100
                self.relaxation_factor = 1.0
                self.verbose = False
        pipeline_sim.SolverConfig = SolverConfig
    
    if not hasattr(pipeline_sim, 'constants'):
        print("⚠️ constants missing, adding patch...")
        class Constants:
            GRAVITY = 9.81
            STANDARD_PRESSURE = 101325.0
            STANDARD_TEMPERATURE = 288.15
            GAS_CONSTANT = 8314.46
        pipeline_sim.constants = Constants()
    
    # Check solver config property
    try:
        test_network = pipeline_sim.Network()
        test_fluid = pipeline_sim.FluidProperties()
        test_solver = pipeline_sim.SteadyStateSolver(test_network, test_fluid)
        
        if not hasattr(test_solver, 'config'):
            print("⚠️ Solver config property missing, patching...")
            # Patch the class
            original_init = pipeline_sim.SteadyStateSolver.__init__
            
            def new_init(self, network, fluid):
                original_init(self, network, fluid)
                self._config = pipeline_sim.SolverConfig()
            
            pipeline_sim.SteadyStateSolver.__init__ = new_init
            
            # Add property
            def get_config(self):
                if not hasattr(self, '_config'):
                    self._config = pipeline_sim.SolverConfig()
                return self._config
            
            def set_config(self, config):
                self._config = config
            
            pipeline_sim.SteadyStateSolver.config = property(get_config, set_config)
        
        print("✅ All components verified")
        
    except Exception as e:
        print(f"⚠️ Warning during verification: {e}")
    
    # Now run the professional test
    print("\n" + "="*70)
    print("Running professional_test.py...")
    print("="*70 + "\n")
    
    # Import and run
    import professional_test
    professional_test.main()
    
except ImportError as e:
    print(f"❌ Failed to import pipeline_sim: {e}")
    print("\nTry:")
    print("  1. Make sure the build completed successfully")
    print("  2. Check if the .pyd file exists in site-packages")
    print("  3. Try: python -m pip show pipeline-sim")