# AI_GENERATED: Pipeline-Sim Python API
"""
Pipeline-Sim: Next-generation petroleum pipeline simulation
"""

__version__ = "0.1.0"

# Import core components from the compiled extension
try:
    # Import everything from the compiled module
    from .pipeline_sim import *
    
    # Also make the ml submodule available
    from . import pipeline_sim
    ml = pipeline_sim.ml
    
    # For convenience, also import commonly used ML classes
    from .pipeline_sim.ml import (
        FeatureExtractor,
        FlowPatternPredictor,
        AnomalyDetector,
        MLOptimizer,
        DataDrivenCorrelation,
        DigitalTwin,
        OptimizationObjective,
        OptimizationConstraints,
        OptimizationResult,
        AnomalyResult,
        EstimatedState,
        Discrepancy
    )
    
except ImportError as e:
    print(f"Warning: C++ extensions not built properly. Error: {e}")
    print("Run 'python setup_complete.py build' to compile the extension.")

# Python-only components (if they exist)
try:
    from .utils import load_network, save_results, plot_network
except ImportError:
    # These Python modules might not exist yet
    pass

try:
    from .correlations import BeggsBrill, HagedornBrown
except ImportError:
    pass

try:
    from .report import generate_report
except ImportError:
    pass

# Define what's available when using "from pipeline_sim import *"
__all__ = [
    # Core classes
    "Network",
    "Node", 
    "Pipe",
    "NodeType",
    "FluidProperties",
    "SteadyStateSolver",
    "TransientSolver",
    "SolverConfig",
    "SolutionResults",
    "FlowPattern",
    "FlowCorrelationResults",
    
    # ML module
    "ml",
    
    # ML classes for convenience
    "FeatureExtractor",
    "FlowPatternPredictor",
    "AnomalyDetector",
    "MLOptimizer",
    "DataDrivenCorrelation",
    "DigitalTwin",
    "OptimizationObjective",
    "OptimizationConstraints",
    "OptimizationResult",
    "AnomalyResult",
    "EstimatedState",
    "Discrepancy",
    
    # Version
    "__version__"
]

# Add Python utility functions if they're available
if 'load_network' in locals():
    __all__.extend(["load_network", "save_results", "plot_network"])
    
if 'BeggsBrill' in locals():
    __all__.extend(["BeggsBrill", "HagedornBrown"])
    
if 'generate_report' in locals():
    __all__.append("generate_report")