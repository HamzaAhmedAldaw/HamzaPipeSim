"""Pipeline-Sim: Next-generation petroleum pipeline simulation"""

__version__ = "0.1.0"

import importlib.util
import os

# Load the _core.pyd with the correct module name
_dir = os.path.dirname(os.path.abspath(__file__))
_pyd = os.path.join(_dir, "_core.cp313-win_amd64.pyd")

spec = importlib.util.spec_from_file_location("pipeline_sim", _pyd)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)

# Import everything
for name in dir(_mod):
    if not name.startswith('_'):
        globals()[name] = getattr(_mod, name)

# Clean up
del importlib, os, spec, _mod, _dir, _pyd, name

# Import Python components
try:
    from .utils import load_network, save_results, plot_network
    from .correlations import BeggsBrill, HagedornBrown
    from .report import generate_report
except ImportError:
    pass