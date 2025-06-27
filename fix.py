#!/usr/bin/env python3
"""
Automatic fix for pipeline_sim import issue
Just run: python auto_fix_import.py
"""

import os
import sys
import site
import shutil

print("Pipeline-Sim Import Auto-Fixer")
print("=" * 60)

# Step 1: Find the installation
print("\n1. Finding pipeline_sim installation...")

pkg_path = None
try:
    import pipeline_sim
    pkg_path = os.path.dirname(pipeline_sim.__file__)
    print(f"   Found at: {pkg_path}")
except Exception as e:
    print(f"   Error during import: {e}")
    
    # Search manually
    for sp in site.getsitepackages():
        # Check egg format
        egg_pattern = os.path.join(sp, "pipeline_sim-0.1.0-py*.egg", "pipeline_sim")
        import glob
        eggs = glob.glob(egg_pattern)
        if eggs:
            pkg_path = eggs[0]
            print(f"   Found in egg at: {pkg_path}")
            break
        
        # Check regular
        regular = os.path.join(sp, "pipeline_sim")
        if os.path.exists(regular):
            pkg_path = regular
            print(f"   Found at: {pkg_path}")
            break

if not pkg_path:
    print("\n✗ Cannot find pipeline_sim installation!")
    print("  Please install it first: python setup_complete.py install")
    sys.exit(1)

# Step 2: Check for compiled extension
print("\n2. Checking for compiled extension...")

extension_file = None
for f in os.listdir(pkg_path):
    if f.endswith(('.pyd', '.so')):
        extension_file = f
        print(f"   Found: {f}")
        break

if not extension_file:
    print("\n✗ No compiled extension found!")
    print("  You need to build it first: python setup_complete.py build")
    sys.exit(1)

# Step 3: Fix __init__.py
print("\n3. Fixing __init__.py...")

init_path = os.path.join(pkg_path, "__init__.py")

# Backup
backup_path = init_path + ".original"
if not os.path.exists(backup_path):
    shutil.copy2(init_path, backup_path)
    print(f"   Created backup: {backup_path}")

# Write fixed version
fixed_init = f'''"""Pipeline-Sim: Next-generation petroleum pipeline simulation"""
__version__ = "0.1.0"

# Direct import from the compiled extension
import os
import importlib.util

_dir = os.path.dirname(__file__)
_ext_path = os.path.join(_dir, "{extension_file}")

if os.path.exists(_ext_path):
    spec = importlib.util.spec_from_file_location("_pipeline_sim_core", _ext_path)
    if spec and spec.loader:
        _core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_core)
        
        # Import everything
        for attr in dir(_core):
            if not attr.startswith('_'):
                globals()[attr] = getattr(_core, attr)
                
        # Handle ML if present
        if hasattr(_core, 'ml'):
            ml = _core.ml
        
        print("Successfully loaded pipeline_sim extension")
else:
    raise ImportError(f"Extension not found at {{_ext_path}}")

# Python components (optional)
try:
    from .utils import *
except ImportError:
    pass

__all__ = [n for n in globals() if not n.startswith('_')]
'''

with open(init_path, 'w') as f:
    f.write(fixed_init)

print("   ✓ Fixed __init__.py")

# Step 4: Test the fix
print("\n4. Testing the fix...")

# Force reload
if 'pipeline_sim' in sys.modules:
    del sys.modules['pipeline_sim']

try:
    import pipeline_sim
    
    # Test imports
    success = True
    for cls in ['Network', 'Node', 'Pipe', 'FluidProperties']:
        if hasattr(pipeline_sim, cls):
            print(f"   ✓ {cls} is available")
        else:
            print(f"   ✗ {cls} is missing")
            success = False
    
    if success:
        print("\n✓ SUCCESS! The import issue has been fixed!")
        print("\nYou can now use:")
        print("  from pipeline_sim import Network, Node, Pipe, FluidProperties, SteadyStateSolver")
        print("\nTry running your test script again.")
    else:
        print("\n⚠ Partial success - some classes are missing")
        
except Exception as e:
    print(f"\n✗ Import still failing: {e}")
    print("\nThe extension file might be corrupted. Try:")
    print("  1. python setup_complete.py clean --all")
    print("  2. python setup_complete.py build")
    print("  3. python setup_complete.py install")

print("\n" + "=" * 60)
input("\nPress Enter to exit...")