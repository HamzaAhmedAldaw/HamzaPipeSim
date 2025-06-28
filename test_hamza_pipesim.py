#!/usr/bin/env python3
"""
Script to find the .pyd file and fix the installation
"""

import os
import shutil
import sys

def find_pyd_files():
    """Search for the compiled .pyd file in common locations"""
    print("Searching for .pyd files...")
    
    search_locations = [
        # Build directories
        r"C:\Users\KIMO STORE\HamzaPipeSim\build",
        r"C:\Users\KIMO STORE\HamzaPipeSim\build\lib.win-amd64-cpython-313",
        r"C:\Users\KIMO STORE\HamzaPipeSim\build\temp.win-amd64-cpython-313",
        
        # Current directory
        os.getcwd(),
        
        # Parent directory
        os.path.dirname(os.getcwd()),
    ]
    
    found_files = []
    
    for location in search_locations:
        if os.path.exists(location):
            print(f"\nChecking: {location}")
            for root, dirs, files in os.walk(location):
                for file in files:
                    if file.endswith('.pyd') and 'pipeline_sim' in file:
                        full_path = os.path.join(root, file)
                        size = os.path.getsize(full_path) / (1024 * 1024)  # MB
                        found_files.append((full_path, size))
                        print(f"  ✓ Found: {file} ({size:.2f} MB)")
    
    return found_files

def get_installation_path():
    """Get the current installation path"""
    try:
        import pipeline_sim
        module_path = pipeline_sim.__file__
        package_dir = os.path.dirname(module_path)
        return package_dir
    except:
        return None

def copy_pyd_to_installation(pyd_path, install_path):
    """Copy the .pyd file to the installation directory"""
    pyd_filename = os.path.basename(pyd_path)
    dest_path = os.path.join(install_path, pyd_filename)
    
    print(f"\nCopying extension file:")
    print(f"  From: {pyd_path}")
    print(f"  To: {dest_path}")
    
    try:
        shutil.copy2(pyd_path, dest_path)
        print("  ✓ Copy successful")
        return True
    except Exception as e:
        print(f"  ✗ Copy failed: {e}")
        return False

def fix_init_py(install_path):
    """Update __init__.py to load the extension properly"""
    init_path = os.path.join(install_path, '__init__.py')
    
    print(f"\nUpdating {init_path}...")
    
    # Find .pyd file in the directory
    pyd_files = [f for f in os.listdir(install_path) if f.endswith('.pyd')]
    
    if not pyd_files:
        print("  ✗ No .pyd file found in installation directory")
        return False
    
    pyd_name = pyd_files[0].replace('.pyd', '')
    
    # Create new __init__.py content
    new_content = f'''"""HamzaPipeSim - Pipeline Simulation Package"""

__version__ = "0.1.0"

# Import from C++ extension
try:
    from .{pyd_name} import *
    print("✓ Successfully loaded C++ extension")
except ImportError as e:
    print(f"Warning: C++ extension not loaded: {{e}}")
    print("Core simulation features will not be available.")

# Import Python utilities
try:
    from . import utils
    from . import correlations  
    from . import report
except ImportError:
    pass
'''
    
    # Backup original
    backup_path = init_path + '.backup'
    if os.path.exists(init_path):
        shutil.copy2(init_path, backup_path)
        print(f"  ✓ Created backup: {backup_path}")
    
    # Write new content
    with open(init_path, 'w') as f:
        f.write(new_content)
    print("  ✓ Updated __init__.py")
    
    return True

def test_import():
    """Test if the import works after fixes"""
    print("\nTesting import after fixes...")
    
    # Remove from sys.modules to force reload
    if 'pipeline_sim' in sys.modules:
        del sys.modules['pipeline_sim']
    
    try:
        import pipeline_sim
        
        # Check for core classes
        required = ['Network', 'NodeType', 'FluidProperties', 'SteadyStateSolver']
        found = []
        
        for cls in required:
            if hasattr(pipeline_sim, cls):
                found.append(cls)
        
        if found:
            print(f"  ✓ Found classes: {', '.join(found)}")
            return True
        else:
            print("  ✗ No C++ classes found")
            return False
            
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False

def main():
    print("="*60)
    print("Pipeline Sim Extension Fixer")
    print("="*60)
    
    # Step 1: Find .pyd files
    pyd_files = find_pyd_files()
    
    if not pyd_files:
        print("\n✗ No .pyd files found!")
        print("\nThe C++ extension needs to be built first.")
        print("Run these commands:")
        print("  cd C:\\Users\\KIMO STORE\\HamzaPipeSim")
        print("  python setup_complete.py build")
        return
    
    # Step 2: Get installation path
    install_path = get_installation_path()
    
    if not install_path:
        print("\n✗ Could not find installation path")
        return
    
    print(f"\nInstallation path: {install_path}")
    
    # Check if .pyd already exists there
    existing_pyd = [f for f in os.listdir(install_path) if f.endswith('.pyd')]
    if existing_pyd:
        print(f"  ℹ Found existing .pyd: {existing_pyd}")
    
    # Step 3: Copy the newest/largest .pyd file
    if pyd_files:
        # Sort by size (largest first)
        pyd_files.sort(key=lambda x: x[1], reverse=True)
        best_pyd = pyd_files[0][0]
        
        if not existing_pyd or input("\nCopy .pyd file to installation? (y/n): ").lower() == 'y':
            if copy_pyd_to_installation(best_pyd, install_path):
                # Step 4: Fix __init__.py
                fix_init_py(install_path)
                
                # Step 5: Test
                if test_import():
                    print("\n✓ SUCCESS! The extension is now properly installed.")
                    print("\nYou may need to restart Python for changes to take effect.")
                else:
                    print("\n⚠ Import test failed. You may need to restart Python.")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()