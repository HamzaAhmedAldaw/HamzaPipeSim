#!/usr/bin/env python3
"""
Fix the __init__.py to properly load the C++ extension
"""

import os
import sys
import shutil

print("Fixing PYD Loading Issue")
print("="*60)

# Find the installation
try:
    import pipeline_sim
    pkg_dir = os.path.dirname(pipeline_sim.__file__)
    print(f"[OK] Found package at: {pkg_dir}")
except ImportError:
    print("[ERROR] Cannot import pipeline_sim")
    sys.exit(1)

# Check for .pyd file
pyd_files = [f for f in os.listdir(pkg_dir) if f.endswith('.pyd')]
if not pyd_files:
    print("[ERROR] No .pyd file found in package directory")
    print("The C++ extension might not have been installed properly")
    sys.exit(1)

pyd_file = pyd_files[0]
print(f"[OK] Found PYD file: {pyd_file}")

# Create new __init__.py content
new_init = '''"""HamzaPipeSim - Pipeline Simulation Package"""

__version__ = "0.1.0"

# Import everything from the C++ extension
import os
import sys

# Get the directory containing this file
_current_dir = os.path.dirname(__file__)

# Find the .pyd file
_pyd_file = None
for _file in os.listdir(_current_dir):
    if _file.endswith('.pyd'):
        _pyd_file = _file
        break

if _pyd_file:
    # Import using the standard method
    try:
        # Try direct import first
        _module_name = _pyd_file[:-4]  # Remove .pyd extension
        
        # Import all symbols from the extension
        import importlib.util
        _spec = importlib.util.spec_from_file_location(
            "pipeline_sim_cpp", 
            os.path.join(_current_dir, _pyd_file)
        )
        _cpp_module = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_cpp_module)
        
        # Copy all public symbols to this namespace
        for _attr in dir(_cpp_module):
            if not _attr.startswith('_'):
                globals()[_attr] = getattr(_cpp_module, _attr)
        
        print("[OK] Successfully loaded C++ extension")
        
    except Exception as e:
        print(f"[ERROR] Failed to load C++ extension: {e}")
        raise
else:
    print("[ERROR] No .pyd file found in package directory")
    raise ImportError("C++ extension not found")

# Import Python submodules
try:
    from . import correlations
    from . import utils
    from . import report
except ImportError:
    pass  # Optional modules

# Clean up temporary variables
del _current_dir, _pyd_file
if '_cpp_module' in globals():
    del _cpp_module
if '_spec' in globals():
    del _spec
if '_module_name' in globals():
    del _module_name
if '_file' in globals():
    del _file
if '_attr' in globals():
    del _attr
'''

# Backup current __init__.py
init_path = os.path.join(pkg_dir, '__init__.py')
backup_path = init_path + '.backup_original'

if not os.path.exists(backup_path):
    shutil.copy2(init_path, backup_path)
    print(f"[OK] Created backup: {backup_path}")

# Write new __init__.py
with open(init_path, 'w', encoding='utf-8') as f:
    f.write(new_init)
print("[OK] Updated __init__.py")

# Test the fix
print("\nTesting the fix...")
# Force reload
if 'pipeline_sim' in sys.modules:
    del sys.modules['pipeline_sim']

try:
    import pipeline_sim
    
    # Check for C++ classes
    required = ['Network', 'NodeType', 'FluidProperties', 'SteadyStateSolver']
    found = []
    
    for cls in required:
        if hasattr(pipeline_sim, cls):
            found.append(cls)
    
    if found:
        print(f"[SUCCESS] Found C++ classes: {', '.join(found)}")
        print("\nThe fix worked! You can now use HamzaPipeSim.")
        
        # Quick test
        try:
            net = pipeline_sim.Network()
            print(f"[OK] Created Network with {net.node_count()} nodes")
        except Exception as e:
            print(f"[WARNING] Could not create Network: {e}")
    else:
        print("[WARNING] C++ classes still not found")
        print("Please close and restart Python, then test again")
        
except Exception as e:
    print(f"[ERROR] Import failed: {e}")
    print("Please restart Python and try again")

print("\n" + "="*60)
print("Fix complete. If classes are not found, restart Python.")