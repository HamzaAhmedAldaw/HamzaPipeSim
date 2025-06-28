# Pipeline-Sim Python API
"""
Pipeline-Sim: Next-generation petroleum pipeline simulation
"""

__version__ = "0.1.0"

# The issue is determining where the compiled extension actually is
# Let's try multiple import strategies

_import_error = None

# Strategy 1: Try importing from the compiled extension with the original name
try:
    from pipeline_sim import *
    _EXTENSION_IMPORTED = True
except ImportError as e:
    _import_error = e
    _EXTENSION_IMPORTED = False

# Strategy 2: If that didn't work, try the egg/build structure
if not _EXTENSION_IMPORTED:
    try:
        # The extension might be compiled as 'pipeline_sim' at the package level
        import pipeline_sim as _core
        
        # Now import everything from it
        for attr in dir(_core):
            if not attr.startswith('_'):
                globals()[attr] = getattr(_core, attr)
        
        _EXTENSION_IMPORTED = True
    except ImportError as e:
        _import_error = e

# Strategy 3: Try importing directly from current directory
if not _EXTENSION_IMPORTED:
    try:
        # Look for any .pyd or .so file in the current directory
        import os
        import importlib.util
        
        current_dir = os.path.dirname(__file__)
        for file in os.listdir(current_dir):
            if file.endswith(('.pyd', '.so')) and file.startswith('pipeline_sim'):
                module_name = file.split('.')[0]
                spec = importlib.util.spec_from_file_location(module_name, os.path.join(current_dir, file))
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Import everything from the module
                    for attr in dir(module):
                        if not attr.startswith('_'):
                            globals()[attr] = getattr(module, attr)
                    
                    _EXTENSION_IMPORTED = True
                    break
    except Exception as e:
        _import_error = e

# If nothing worked, show helpful error
if not _EXTENSION_IMPORTED:
    print(f"Warning: C++ extensions not available. Error: {_import_error}")
    print("The extension module may not be installed correctly.")
    print("Try:")
    print("  1. python setup_complete.py clean --all")
    print("  2. python setup_complete.py build") 
    print("  3. python setup_complete.py install")
    
    # Create dummy classes so imports don't completely fail
    class DummyClass:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("C++ extension not available. Please build and install the extension.")
    
    # Basic classes that should exist
    Network = DummyClass
    Node = DummyClass
    Pipe = DummyClass
    FluidProperties = DummyClass
    SteadyStateSolver = DummyClass
    TransientSolver = DummyClass
    
    class NodeType:
        JUNCTION = 0
        SOURCE = 1
        SINK = 2
        PUMP = 3
        COMPRESSOR = 4
        VALVE = 5
        SEPARATOR = 6
        HEAT_EXCHANGER = 7

# Handle ML submodule
if _EXTENSION_IMPORTED and 'ml' not in globals():
    # Create ml submodule if it doesn't exist but ML classes do
    import types
    ml = types.ModuleType('ml')
    
    # List of expected ML classes
    ml_classes = [
        'FeatureExtractor', 'FlowPatternPredictor', 'AnomalyDetector',
        'MLOptimizer', 'DataDrivenCorrelation', 'DigitalTwin',
        'OptimizationObjective', 'OptimizationConstraints', 'OptimizationResult',
        'AnomalyResult', 'EstimatedState', 'Discrepancy'
    ]
    
    # Move ML classes to ml submodule if they exist
    ml_found = False
    for cls_name in ml_classes:
        if cls_name in globals():
            setattr(ml, cls_name, globals()[cls_name])
            ml_found = True
    
    if ml_found:
        globals()['ml'] = ml

# Python-only components (if they exist)
try:
    from .utils import load_network, save_results, plot_network
except ImportError:
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
__all__ = []

# Add all imported C++ classes
if _EXTENSION_IMPORTED:
    __all__ = [name for name in globals() if not name.startswith('_') and name != 'os' and name != 'importlib']
else:
    # Minimum set for compatibility
    __all__ = [
        "Network", "Node", "Pipe", "NodeType", "FluidProperties",
        "SteadyStateSolver", "TransientSolver", "__version__"
    ]