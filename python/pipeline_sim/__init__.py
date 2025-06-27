# AI_GENERATED: Pipeline-Sim Python API
"""
Pipeline-Sim: Next-generation petroleum pipeline simulation
"""

__version__ = "0.1.0"

# Import core components
try:
    from ._core import (
        Network,
        Node,
        Pipe,
        NodeType,
        FluidProperties,
        SteadyStateSolver,
        TransientSolver,
    )
except ImportError:
    print("Warning: C++ extensions not built. Run 'pip install -e .' to build.")

# Python-only components
from .utils import load_network, save_results, plot_network
from .correlations import BeggsBrill, HagedornBrown
from .report import generate_report

__all__ = [
    "Network",
    "Node",
    "Pipe",
    "NodeType",
    "FluidProperties",
    "SteadyStateSolver",
    "TransientSolver",
    "load_network",
    "save_results",
    "plot_network",
    "BeggsBrill",
    "HagedornBrown",
    "generate_report",
]