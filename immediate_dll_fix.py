#!/usr/bin/env python3
"""
Immediate fix for DLL loading error
"""

import os
import sys

print("Fixing HamzaPipeSim DLL Loading Error")
print("="*60)

# Step 1: Remove the problematic stub loader
stub_path = r"C:\Users\KIMO STORE\miniconda3\Lib\site-packages\pipeline_sim.py"
if os.path.exists(stub_path):
    print(f"Removing problematic stub loader: {stub_path}")
    try:
        os.remove(stub_path)
        print("✓ Removed stub loader")
        
        # Also remove .pyc if exists
        pyc_path = stub_path + 'c'
        if os.path.exists(pyc_path):
            os.remove(pyc_path)
    except Exception as e:
        print(f"✗ Could not remove stub: {e}")
        print("  Try manually: del", stub_path)

# Step 2: Test direct import from egg
print("\nTesting direct import from egg...")

egg_path = r"C:\Users\KIMO STORE\miniconda3\Lib\site-packages\pipeline_sim-0.1.0-py3.13-win-amd64.egg"

# Clear any cached imports
if 'pipeline_sim' in sys.modules:
    del sys.modules['pipeline_sim']

# Add egg to path
sys.path.insert(0, egg_path)

try:
    import pipeline_sim
    
    # Test if C++ classes are available
    if hasattr(pipeline_sim, 'Network'):
        net = pipeline_sim.Network()
        print(f"✓ SUCCESS! Network class is available")
        print(f"  Created network with {net.node_count()} nodes")
        
        # Test other classes
        classes = ['NodeType', 'FluidProperties', 'SteadyStateSolver']
        for cls in classes:
            if hasattr(pipeline_sim, cls):
                print(f"  ✓ {cls} is available")
        
        print("\n" + "="*60)
        print("FIX SUCCESSFUL!")
        print("="*60)
        print("\nTo use HamzaPipeSim in your scripts, add this at the beginning:")
        print(f'import sys')
        print(f'sys.path.insert(0, r"{egg_path}")')
        print('import pipeline_sim')
        print("\nThen you can run:")
        print("  python complete_demo.py")
        
    else:
        print("✗ Network class not found")
        attrs = [x for x in dir(pipeline_sim) if not x.startswith('_')]
        print(f"Available: {attrs}")
        
except Exception as e:
    print(f"✗ Import error: {e}")
    
    # Try alternative: Install Visual C++ Redistributable
    print("\nThe error might be due to missing Visual C++ Runtime")
    print("Please install:")
    print("  https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print("\nAfter installing, restart your computer and try again.")

# Step 3: Create a permanent fix script
print("\nCreating permanent import fix...")

fix_content = '''#!/usr/bin/env python3
"""
Import fix for HamzaPipeSim
Add this to the beginning of your scripts
"""

import sys
import os

# Add egg to Python path
egg_path = r"C:\\Users\\KIMO STORE\\miniconda3\\Lib\\site-packages\\pipeline_sim-0.1.0-py3.13-win-amd64.egg"
if egg_path not in sys.path:
    sys.path.insert(0, egg_path)

# Now you can import pipeline_sim
import pipeline_sim

# Optional: Print success message
if hasattr(pipeline_sim, 'Network'):
    print("✓ HamzaPipeSim loaded successfully")
'''

with open("fix_import.py", 'w') as f:
    f.write(fix_content)

print("✓ Created fix_import.py")
print("\nIn your scripts, use:")
print("  import fix_import")
print("  # Now use pipeline_sim normally")
print("  net = pipeline_sim.Network()")

print("\n" + "="*60)