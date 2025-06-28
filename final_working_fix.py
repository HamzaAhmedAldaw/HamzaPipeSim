# verify_and_rebuild.py
"""
Verify bindings.cpp was fixed and do a clean rebuild
"""
import os
import shutil
import subprocess
import sys

def verify_and_rebuild():
    print("Verifying and Rebuilding HamzaPipeSim")
    print("=" * 60)
    
    # Check bindings.cpp
    bindings_path = r"C:\Users\KIMO STORE\HamzaPipeSim\python\src\bindings.cpp"
    
    print("1. Checking bindings.cpp...")
    if os.path.exists(bindings_path):
        with open(bindings_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if FlowCorrelation is registered before DataDrivenCorrelation
        flow_pos = content.find('py::class_<FlowCorrelation')
        data_pos = content.find('py::class_<DataDrivenCorrelation')
        
        if flow_pos > 0 and data_pos > 0:
            if flow_pos < data_pos:
                print("✓ Bindings are correctly ordered (FlowCorrelation before DataDrivenCorrelation)")
            else:
                print("✗ Bindings are NOT correctly ordered!")
                return False
        else:
            print("✗ Could not find binding definitions")
    else:
        print(f"✗ bindings.cpp not found at: {bindings_path}")
        return False
    
    # Clean build directory
    build_dir = r"C:\Users\KIMO STORE\HamzaPipeSim\build"
    
    print("\n2. Cleaning build directory...")
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir, ignore_errors=True)
        print("✓ Removed old build directory")
    
    # Rebuild
    print("\n3. Rebuilding with fixed bindings...")
    os.chdir(r"C:\Users\KIMO STORE\HamzaPipeSim")
    
    result = subprocess.run([sys.executable, "setup_complete.py", "build"], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Build completed successfully")
        
        # Find the new .pyd
        pyd_path = None
        for root, dirs, files in os.walk(build_dir):
            for file in files:
                if file.endswith('.pyd'):
                    pyd_path = os.path.join(root, file)
                    print(f"✓ Created: {pyd_path}")
                    print(f"  Size: {os.path.getsize(pyd_path) / 1024:.1f} KB")
                    break
        
        if not pyd_path:
            print("✗ No .pyd file found after build")
            return False
            
    else:
        print(f"✗ Build failed: {result.stderr}")
        return False
    
    # Test the new .pyd directly
    print("\n4. Testing the newly built .pyd...")
    
    test_code = f'''
import importlib.util
spec = importlib.util.spec_from_file_location("pipeline_sim", r"{pyd_path}")
module = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(module)
    print("✓ Module loaded successfully")
    if hasattr(module, 'Network'):
        net = module.Network()
        print(f"✓ Network class works! Created network with {{net.node_count()}} nodes")
    else:
        attrs = [a for a in dir(module) if not a.startswith('_')]
        print(f"✗ Network not found. Available: {{attrs}}")
except Exception as e:
    print(f"✗ Load error: {{e}}")
'''
    
    result = subprocess.run([sys.executable, "-c", test_code], 
                          capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Error: {result.stderr}")
    
    if "Network class works" in result.stdout:
        print("\n5. Installing the fixed .pyd...")
        
        # Copy to installation
        egg_path = r"C:\Users\KIMO STORE\miniconda3\Lib\site-packages\pipeline_sim-0.1.0-py3.13-win-amd64.egg"
        dest_pyd = os.path.join(egg_path, "pipeline_sim.pyd")
        
        shutil.copy2(pyd_path, dest_pyd)
        print(f"✓ Copied to: {dest_pyd}")
        
        # Also copy with tagged name
        dest_tagged = os.path.join(egg_path, os.path.basename(pyd_path))
        shutil.copy2(pyd_path, dest_tagged)
        
        print("\n✓ Installation complete!")
        print("\nTo use HamzaPipeSim:")
        print("  import sys")
        print('  sys.path.insert(0, r"C:\\Users\\KIMO STORE\\miniconda3\\Lib\\site-packages\\pipeline_sim-0.1.0-py3.13-win-amd64.egg")')
        print("  import pipeline_sim")
        print("  net = pipeline_sim.Network()")
        
        return True
    else:
        print("\n✗ The build still has issues. Let's check what went wrong.")
        return False

if __name__ == "__main__":
    verify_and_rebuild()