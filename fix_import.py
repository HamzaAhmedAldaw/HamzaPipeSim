import sys
import os
import pathlib
import importlib.util

print("=" * 60)
print("DIAGNOSTIC REPORT FOR PIPELINE_SIM")
print("=" * 60)

# 1. Check current directory
print(f"\n1. Current directory: {os.getcwd()}")

# 2. Find all .pyd files
print("\n2. Looking for compiled extensions (.pyd files):")
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".pyd"):
            full_path = os.path.join(root, file)
            print(f"   Found: {full_path}")
            print(f"   Size: {os.path.getsize(full_path)} bytes")

# 3. Check Python path
print("\n3. Python sys.path:")
for i, p in enumerate(sys.path[:5]):  # First 5 entries
    print(f"   [{i}] {p}")

# 4. Try to import pipeline_sim
print("\n4. Attempting to import pipeline_sim:")
try:
    import pipeline_sim
    print("   SUCCESS: pipeline_sim imported")
    print(f"   Module file: {pipeline_sim.__file__}")
    print(f"   Module attributes: {[attr for attr in dir(pipeline_sim) if not attr.startswith('_')]}")
except Exception as e:
    print(f"   FAILED: {type(e).__name__}: {e}")

# 5. Try to load .pyd directly
print("\n5. Trying to load .pyd files directly:")
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".pyd") and "pipeline_sim" in file:
            full_path = os.path.abspath(os.path.join(root, file))
            module_name = file.replace(".pyd", "").replace(".cp313-win_amd64", "")
            print(f"\n   Attempting to load: {file}")
            print(f"   Module name: {module_name}")
            
            try:
                spec = importlib.util.spec_from_file_location(module_name, full_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                print(f"   SUCCESS: Loaded {module_name}")
                print(f"   Attributes: {[attr for attr in dir(module) if not attr.startswith('_')][:10]}...")
            except Exception as e:
                print(f"   FAILED: {type(e).__name__}: {e}")

# 6. Check if there's a naming conflict
print("\n6. Checking for naming conflicts:")
if os.path.exists("pipeline_sim.py"):
    print("   WARNING: Found pipeline_sim.py file which may conflict!")
if os.path.exists("python/pipeline_sim.py"):
    print("   WARNING: Found python/pipeline_sim.py file which may conflict!")

print("\n" + "=" * 60)