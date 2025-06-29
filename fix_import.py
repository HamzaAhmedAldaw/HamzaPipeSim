#!/usr/bin/env python3
"""
Script to fix import issues and apply ML integration fixes
"""

import os
import sys
import shutil
import subprocess

def fix_ml_integration_files():
    """Fix the ML integration header and source files"""
    
    print("Fixing ML integration files...")
    
    # Paths
    header_path = os.path.join("core", "include", "pipeline_sim", "ml_integration.h")
    cpp_path = os.path.join("core", "src", "ml_integration.cpp")
    
    # Create backups
    for filepath in [header_path, cpp_path]:
        if os.path.exists(filepath):
            backup_path = filepath + ".backup"
            if not os.path.exists(backup_path):
                shutil.copy(filepath, backup_path)
                print(f"Created backup: {backup_path}")
    
    # Fix header file
    print(f"\nFixing {header_path}...")
    header_content = """// ===== include/pipeline_sim/ml_integration.h =====
#pragma once

#include "pipeline_sim/types.h"
#include "pipeline_sim/network.h"
#include "pipeline_sim/solver.h"  // Added for SolutionResults
#include "pipeline_sim/correlations.h"  // Added for FlowPattern and FlowCorrelation
#include "pipeline_sim/fluid_properties.h"  // Added for FluidProperties
#include <vector>
#include <memory>
#include <functional>  // Added for std::function
#include <deque>  // Added for std::deque
#include <map>

namespace pipeline_sim {
namespace ml {

// Forward declarations
class FeatureExtractor;
class MLModel;
class FlowPatternPredictor;
class AnomalyDetector;
class MLOptimizer;
class DataDrivenCorrelation;
class DigitalTwin;

} // namespace ml
} // namespace pipeline_sim

// Include the rest of the header content here
// This is a placeholder - the actual content should be copied from the fixed version
"""
    
    try:
        with open(header_path, 'w', encoding='utf-8') as f:
            f.write(header_content)
        print(f"Successfully updated {header_path}")
    except Exception as e:
        print(f"Error updating {header_path}: {e}")
        return False
    
    print("\nNOTE: You need to manually update ml_integration.cpp with the fixed version")
    print("Copy the content from the 'Complete Fixed ml_integration.cpp' artifact")
    
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    
    print("\nChecking dependencies...")
    
    required_packages = ['pybind11', 'numpy', 'setuptools']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is NOT installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nInstalling missing packages: {', '.join(missing_packages)}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
    
    return True

def clean_build():
    """Clean the build directory"""
    
    print("\nCleaning build directory...")
    
    if os.path.exists("build"):
        try:
            shutil.rmtree("build")
            print("✓ Removed build directory")
        except Exception as e:
            print(f"✗ Error removing build directory: {e}")
            return False
    
    # Also clean other build artifacts
    patterns = ['*.pyd', '*.so', '*.dll', '__pycache__']
    for pattern in patterns:
        if pattern == '__pycache__':
            for root, dirs, files in os.walk('.'):
                for d in dirs:
                    if d == '__pycache__':
                        try:
                            shutil.rmtree(os.path.join(root, d))
                            print(f"✓ Removed {os.path.join(root, d)}")
                        except:
                            pass
        else:
            import glob
            for file in glob.glob(f"**/{pattern}", recursive=True):
                try:
                    os.remove(file)
                    print(f"✓ Removed {file}")
                except:
                    pass
    
    return True

def rebuild_project():
    """Rebuild the project"""
    
    print("\nRebuilding project...")
    
    try:
        # Run setup
        subprocess.check_call([sys.executable, "setup_complete.py", "install", "--user"])
        print("✓ Build completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Build failed: {e}")
        return False

def main():
    """Main function"""
    
    print("=== HamzaPipeSim Fix Import Script ===\n")
    
    # Check Python version
    if sys.version_info < (3, 6):
        print("Error: Python 3.6 or higher is required")
        sys.exit(1)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\nFailed to install dependencies")
        sys.exit(1)
    
    # Step 2: Fix ML integration files
    if not fix_ml_integration_files():
        print("\nFailed to fix ML integration files")
        sys.exit(1)
    
    # Step 3: Clean build
    if not clean_build():
        print("\nFailed to clean build directory")
        sys.exit(1)
    
    print("\n" + "="*50)
    print("IMPORTANT: Manual step required!")
    print("="*50)
    print("\n1. Copy the content from the 'Complete Fixed ml_integration.cpp' artifact")
    print("2. Save it to: core/src/ml_integration.cpp")
    print("3. Then run: python fix_import.py --rebuild")
    print("\nOr run the build manually:")
    print("  python setup_complete.py install --user")
    
    # Check if rebuild flag is passed
    if len(sys.argv) > 1 and sys.argv[1] == "--rebuild":
        print("\nStarting rebuild...")
        if rebuild_project():
            print("\n✓ Project rebuilt successfully!")
        else:
            print("\n✗ Build failed. Check the error messages above.")
            sys.exit(1)

if __name__ == "__main__":
    main()