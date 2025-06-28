# AI_GENERATED: Pipeline-Sim Python API
"""
Pipeline-Sim: Next-generation petroleum pipeline simulation
"""

__version__ = "0.1.0"

import os
import sys
import importlib.util
from pathlib import Path

# Try to load the compiled extension
_loaded = False

# Method 1: Try direct import (if installed properly)
try:
    from pipeline_sim import *
    _loaded = True
except ImportError:
    pass

# Method 2: Try to load .pyd file directly
if not _loaded:
    # Look for .pyd file in various locations
    possible_paths = [
        Path(__file__).parent,  # Same directory as __init__.py
        Path(__file__).parent.parent,  # Parent directory
        Path(__file__).parent.parent.parent / "build" / "lib.win-amd64-cpython-313",  # Build directory
    ]
    
    for base_path in possible_paths:
        if not base_path.exists():
            continue
            
        # Look for .pyd files
        for pyd_file in base_path.glob("pipeline_sim*.pyd"):
            try:
                # Load the module from the .pyd file
                spec = importlib.util.spec_from_file_location("pipeline_sim_core", pyd_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules["pipeline_sim_core"] = module
                    spec.loader.exec_module(module)
                    
                    # Import all symbols
                    for name in dir(module):
                        if not name.startswith('_'):
                            globals()[name] = getattr(module, name)
                    
                    _loaded = True
                    break
            except Exception as e:
                continue
        
        if _loaded:
            break

if not _loaded:
    raise ImportError(
        "Could not load Pipeline-Sim C++ extension. "
        "Make sure it's properly built with: python setup_complete.py build_ext"
    )

# Python-only components (import after core is loaded)
try:
    from .utils import load_network, save_results, plot_network
    from .correlations import BeggsBrill, HagedornBrown
    from .report import generate_report
except ImportError:
    # These might not exist yet
    pass

# Define what's available
__all__ = [
    "Network",
    "Node", 
    "Pipe",
    "NodeType",
    "FluidProperties",
    "Solver",
    "SteadyStateSolver",
    "SolutionResults",
    "SolverConfig",
    "get_version",
]

# Add version function if not available
if 'get_version' not in globals():
    def get_version():
        return __version__