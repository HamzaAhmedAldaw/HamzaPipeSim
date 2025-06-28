#!/usr/bin/env python3
"""
Diagnose why the C++ extension isn't loading
"""

import os
import sys

print("Diagnosing PYD Loading Issue")
print("="*60)

# Step 1: Import pipeline_sim and check its location
print("\n1. Checking pipeline_sim location...")
try:
    import pipeline_sim
    print(f"   [OK] Imported from: {pipeline_sim.__file__}")
    pkg_dir = os.path.dirname(pipeline_sim.__file__)
    print(f"   Package directory: {pkg_dir}")
except ImportError as e:
    print(f"   [ERROR] Cannot import: {e}")
    sys.exit(1)

# Step 2: List files in the package directory
print("\n2. Files in package directory:")
if os.path.exists(pkg_dir):
    files = os.listdir(pkg_dir)
    for f in sorted(files):
        size = os.path.getsize(os.path.join(pkg_dir, f)) / 1024  # KB
        print(f"   - {f} ({size:.1f} KB)")
        
    # Check for .pyd files
    pyd_files = [f for f in files if f.endswith('.pyd')]
    if pyd_files:
        print(f"\n   [OK] Found PYD file(s): {pyd_files}")
    else:
        print("\n   [ERROR] No .pyd files found!")

# Step 3: Check __init__.py content
print("\n3. Checking __init__.py content...")
init_path = os.path.join(pkg_dir, '__init__.py')
if os.path.exists(init_path):
    with open(init_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    print("   First 500 characters:")
    print("   " + "-"*50)
    print("   " + content[:500].replace('\n', '\n   '))
    print("   " + "-"*50)
    
    # Check for import attempts
    if '.pyd' in content or 'import' in content:
        print("   [OK] __init__.py contains import statements")
    else:
        print("   [WARNING] __init__.py might not be importing the extension")

# Step 4: Check what's actually in the module
print("\n4. Checking module contents...")
attrs = dir(pipeline_sim)
public_attrs = [a for a in attrs if not a.startswith('_')]
print(f"   Found {len(public_attrs)} public attributes:")
for attr in sorted(public_attrs)[:10]:  # Show first 10
    print(f"   - {attr}")
if len(public_attrs) > 10:
    print(f"   ... and {len(public_attrs) - 10} more")

# Step 5: Try to manually load the .pyd
print("\n5. Attempting manual PYD load...")
if pyd_files:
    pyd_path = os.path.join(pkg_dir, pyd_files[0])
    print(f"   Trying to load: {pyd_path}")
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("test_pyd", pyd_path)
        test_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_module)
        
        print("   [OK] PYD loaded successfully!")
        print("   Contents:")
        test_attrs = [a for a in dir(test_module) if not a.startswith('_')]
        for attr in test_attrs[:10]:
            print(f"   - {attr}")
        
        # Check for Network class
        if hasattr(test_module, 'Network'):
            print("\n   [OK] Network class found in PYD!")
            print("   The issue is with __init__.py loading")
        
    except Exception as e:
        print(f"   [ERROR] Failed to load PYD: {e}")

# Step 6: Check for the stub loader
print("\n6. Checking for stub loader...")
stub_path = os.path.join(os.path.dirname(pkg_dir), 'pipeline_sim.py')
if os.path.exists(stub_path):
    print(f"   [WARNING] Found stub loader at: {stub_path}")
    print("   This might be interfering with the import")
else:
    print("   [OK] No stub loader found")

print("\n" + "="*60)
print("Diagnosis complete. Check the output above for issues.")