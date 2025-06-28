#!/usr/bin/env python3
"""
Complete clean rebuild script for HamzaPipeSim
Handles spaces in paths and does everything in Python
"""

import os
import sys
import shutil
import subprocess
import site

def run_command(cmd):
    """Run a command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def clean_uninstall():
    """Uninstall pipeline_sim manually"""
    print("Step 1: Cleaning previous installations...")
    
    # Method 1: Try pip uninstall using Python module
    print("  Trying pip uninstall...")
    success, out, err = run_command([sys.executable, "-m", "pip", "uninstall", "pipeline_sim", "-y"])
    if success:
        print("  ✓ Pip uninstall successful")
    else:
        print("  ⚠ Pip uninstall failed, trying manual removal...")
    
    # Method 2: Manual removal from site-packages
    removed = False
    for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
        if not os.path.exists(site_dir):
            continue
            
        # Look for pipeline_sim installations
        for item in os.listdir(site_dir):
            if 'pipeline_sim' in item:
                item_path = os.path.join(site_dir, item)
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                    print(f"  ✓ Removed: {item}")
                    removed = True
                except Exception as e:
                    print(f"  ⚠ Could not remove {item}: {e}")
    
    if removed:
        print("  ✓ Manual cleanup complete")
    else:
        print("  ℹ No existing installations found")

def clean_build_artifacts():
    """Remove all build artifacts"""
    print("\nStep 2: Cleaning build artifacts...")
    
    artifacts = ['build', 'dist', '*.egg-info', 'python/*.egg-info', 
                 '__pycache__', '.pytest_cache', '*.pyd']
    
    for pattern in artifacts:
        if '*' in pattern:
            # Handle wildcards
            import glob
            for path in glob.glob(pattern):
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
                    print(f"  ✓ Removed: {path}")
                except Exception as e:
                    print(f"  ⚠ Could not remove {path}: {e}")
        else:
            # Direct path
            if os.path.exists(pattern):
                try:
                    if os.path.isdir(pattern):
                        shutil.rmtree(pattern)
                    else:
                        os.remove(pattern)
                    print(f"  ✓ Removed: {pattern}")
                except Exception as e:
                    print(f"  ⚠ Could not remove {pattern}: {e}")
    
    print("  ✓ Build artifacts cleaned")

def build_extension():
    """Build the C++ extension"""
    print("\nStep 3: Building C++ extension...")
    
    # Run build
    print("  Running: python setup_complete.py build")
    success, out, err = run_command([sys.executable, "setup_complete.py", "build"])
    
    if success:
        print("  ✓ Build completed successfully")
        
        # Check for .pyd file
        pyd_path = os.path.join("build", "lib.win-amd64-cpython-313", "pipeline_sim.cp313-win_amd64.pyd")
        if os.path.exists(pyd_path):
            size = os.path.getsize(pyd_path) / 1024  # KB
            print(f"  ✓ Created .pyd file: {size:.0f} KB")
        else:
            print("  ⚠ Warning: .pyd file not found at expected location")
    else:
        print("  ✗ Build failed!")
        print("Error:", err)
        return False
    
    return True

def install_package():
    """Install the package using pip"""
    print("\nStep 4: Installing package...")
    
    # Try pip install
    print("  Running: pip install .")
    success, out, err = run_command([sys.executable, "-m", "pip", "install", "."])
    
    if success:
        print("  ✓ Installation successful")
    else:
        print("  ✗ Installation failed!")
        print("  Trying alternative method...")
        
        # Alternative: direct copy
        success = install_direct()
    
    return success

def install_direct():
    """Direct installation by copying files"""
    print("\n  Attempting direct installation...")
    
    # Find build output
    build_lib = os.path.join("build", "lib.win-amd64-cpython-313")
    if not os.path.exists(build_lib):
        print("  ✗ Build output not found")
        return False
    
    # Find target directory
    target_dir = None
    for site_dir in site.getsitepackages():
        if os.path.exists(site_dir) and 'site-packages' in site_dir:
            target_dir = os.path.join(site_dir, "pipeline_sim")
            break
    
    if not target_dir:
        print("  ✗ Could not find site-packages directory")
        return False
    
    # Copy files
    try:
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        shutil.copytree(os.path.join(build_lib, "pipeline_sim"), target_dir)
        
        # Also copy the .pyd if it's in build root
        pyd_src = os.path.join(build_lib, "pipeline_sim.cp313-win_amd64.pyd")
        if os.path.exists(pyd_src):
            shutil.copy2(pyd_src, target_dir)
        
        print(f"  ✓ Copied files to: {target_dir}")
        return True
    except Exception as e:
        print(f"  ✗ Copy failed: {e}")
        return False

def test_installation():
    """Test the installation"""
    print("\nStep 5: Testing installation...")
    
    # Force reload
    if 'pipeline_sim' in sys.modules:
        del sys.modules['pipeline_sim']
    
    try:
        import pipeline_sim
        print("  ✓ Import successful")
        
        # Check for C++ classes
        classes = ['Network', 'NodeType', 'FluidProperties', 'SteadyStateSolver']
        found = []
        for cls in classes:
            if hasattr(pipeline_sim, cls):
                found.append(cls)
        
        if found:
            print(f"  ✓ Found C++ classes: {', '.join(found)}")
            
            # Try to create network
            net = pipeline_sim.Network()
            print(f"  ✓ Created Network with {net.node_count()} nodes")
            
            return True
        else:
            print("  ✗ C++ classes not found")
            attrs = [a for a in dir(pipeline_sim) if not a.startswith('_')]
            print(f"  Available: {', '.join(attrs[:10])}")
            return False
            
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False

def main():
    """Main rebuild process"""
    print("="*60)
    print("HamzaPipeSim Complete Clean Rebuild")
    print("="*60)
    
    # Change to project directory
    os.chdir(r"C:\Users\KIMO STORE\HamzaPipeSim")
    print(f"Working directory: {os.getcwd()}")
    
    # Step 1: Clean
    clean_uninstall()
    clean_build_artifacts()
    
    # Step 2: Build
    if not build_extension():
        print("\n✗ Build failed. Please check error messages above.")
        return 1
    
    # Step 3: Install
    if not install_package():
        print("\n✗ Installation failed. Please check error messages above.")
        return 1
    
    # Step 4: Test
    if test_installation():
        print("\n" + "="*60)
        print("✓ SUCCESS! HamzaPipeSim is ready to use!")
        print("="*60)
        return 0
    else:
        print("\n" + "="*60)
        print("⚠ Installation complete but tests failed.")
        print("Try restarting Python and importing again.")
        print("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main())