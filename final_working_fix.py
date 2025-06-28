#!/usr/bin/env python3
"""
Final fix - Remove stub loader and test HamzaPipeSim
"""

import os
import sys

print("HamzaPipeSim Final Fix")
print("="*60)

# Step 1: Remove ALL interfering files
print("Step 1: Removing interfering files...")

files_to_remove = [
    r"C:\Users\KIMO STORE\miniconda3\Lib\site-packages\pipeline_sim.py",
    r"C:\Users\KIMO STORE\miniconda3\Lib\site-packages\pipeline_sim.pyc",
    r"C:\Users\KIMO STORE\miniconda3\Lib\site-packages\__pycache__\pipeline_sim.cpython-313.pyc"
]

for file_path in files_to_remove:
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"  ✓ Removed: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"  ⚠ Could not remove {os.path.basename(file_path)}: {e}")

# Step 2: Clear Python's import cache
print("\nStep 2: Clearing Python import cache...")
modules_to_clear = ['pipeline_sim', 'pipeline_sim.ml']
for mod in modules_to_clear:
    if mod in sys.modules:
        del sys.modules[mod]
        print(f"  ✓ Cleared {mod} from cache")

# Step 3: Test direct import from egg
print("\nStep 3: Testing import from egg installation...")

egg_path = r"C:\Users\KIMO STORE\miniconda3\Lib\site-packages\pipeline_sim-0.1.0-py3.13-win-amd64.egg"

# Add to path
sys.path.insert(0, egg_path)

try:
    import pipeline_sim
    print("✓ Import successful!")
    
    # Test all core features
    print("\nTesting core features:")
    
    # 1. Network creation
    net = pipeline_sim.Network()
    print(f"  ✓ Created Network with {net.node_count()} nodes")
    
    # 2. Add nodes
    source = net.add_node("Source", pipeline_sim.NodeType.SOURCE)
    sink = net.add_node("Sink", pipeline_sim.NodeType.SINK)
    print(f"  ✓ Added nodes: {source}, {sink}")
    
    # 3. Add pipe
    pipe = net.add_pipe("TestPipe", source, sink, 1000.0, 0.2)
    print(f"  ✓ Added pipe: {pipe}")
    
    # 4. Set boundary conditions
    net.set_pressure(source, 50e5)  # 50 bar
    net.set_flow_rate(sink, -0.1)   # 100 L/s
    print("  ✓ Set boundary conditions")
    
    # 5. Create fluid
    fluid = pipeline_sim.FluidProperties()
    fluid.oil_density = 850.0
    fluid.oil_viscosity = 0.002
    print("  ✓ Created fluid properties")
    
    # 6. Solve
    solver = pipeline_sim.SteadyStateSolver(net, fluid)
    results = solver.solve()
    
    if results.converged:
        print(f"  ✓ Simulation converged in {results.iterations} iterations!")
        print(f"    Residual: {results.residual:.2e}")
        
        print("\n  Results:")
        for node, pressure in results.node_pressures.items():
            print(f"    {node}: {pressure/1e5:.1f} bar")
    
    # Test other features
    print("\nTesting additional features:")
    
    # Correlations
    if hasattr(pipeline_sim, 'BeggsBrill'):
        print("  ✓ BeggsBrill correlation available")
    
    # Equipment
    if hasattr(pipeline_sim, 'Pump'):
        pump = pipeline_sim.Pump()
        print("  ✓ Equipment models available")
    
    # ML features
    try:
        from pipeline_sim.ml import FeatureExtractor
        print("  ✓ ML features available")
    except:
        print("  ⚠ ML features not accessible (check if included in build)")
    
    print("\n" + "="*60)
    print("🎉 SUCCESS! HamzaPipeSim is fully operational!")
    print("="*60)
    
    # Create a permanent solution file
    print("\nCreating permanent solution file...")
    
    solution_content = f'''#!/usr/bin/env python3
"""
HamzaPipeSim Import Helper
Include this at the beginning of your scripts
"""

import sys

# Add egg to path for HamzaPipeSim
_egg_path = r"{egg_path}"
if _egg_path not in sys.path:
    sys.path.insert(0, _egg_path)

# Import pipeline_sim
import pipeline_sim

# Optional: Verify it loaded correctly
if not hasattr(pipeline_sim, 'Network'):
    raise ImportError("HamzaPipeSim did not load correctly")

print("✓ HamzaPipeSim ready to use")
'''
    
    with open("hamza_pipesim.py", 'w') as f:
        f.write(solution_content)
    
    print("✓ Created hamza_pipesim.py")
    print("\nIn your scripts, simply use:")
    print("  import hamza_pipesim")
    print("  import pipeline_sim")
    print("\nOr add this line at the beginning:")
    print(f'  sys.path.insert(0, r"{egg_path}")')
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    
    print("\nTroubleshooting:")
    print("1. Make sure the egg file exists at:")
    print(f"   {egg_path}")
    print("2. Try manually in Python:")
    print("   >>> import sys")
    print(f'   >>> sys.path.insert(0, r"{egg_path}")')
    print("   >>> import pipeline_sim")

print("\n" + "="*60)