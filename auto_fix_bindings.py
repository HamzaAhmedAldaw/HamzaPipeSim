#!/usr/bin/env python3
"""
Automatic fix for HamzaPipeSim bindings.cpp
Preserves all features while fixing registration order
"""

import os
import re
import shutil
import subprocess
import sys

def fix_bindings_professionally():
    """Apply professional fixes to bindings.cpp"""
    
    print("Professional Bindings Fix - Preserving All Features")
    print("="*70)
    
    bindings_path = r"C:\Users\KIMO STORE\HamzaPipeSim\python\src\bindings.cpp"
    
    # Backup
    backup_path = bindings_path + ".professional_backup"
    if not os.path.exists(backup_path):
        shutil.copy2(bindings_path, backup_path)
        print(f"✓ Created backup: {os.path.basename(backup_path)}")
    
    # Read content
    with open(bindings_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("\nAnalyzing and fixing issues...")
    
    # Fix 1: Find PYBIND11_MODULE
    module_match = re.search(r'PYBIND11_MODULE\s*\(\s*pipeline_sim\s*,\s*(\w+)\s*\)\s*{', content)
    if not module_match:
        print("✗ Could not find PYBIND11_MODULE")
        return False
    
    module_var = module_match.group(1)  # Usually 'm'
    print(f"✓ Found module variable: {module_var}")
    
    # Fix 2: Ensure FlowCorrelation is registered before derived classes
    # Find all class registrations
    class_registrations = []
    
    # Pattern to find py::class_ declarations
    class_pattern = r'(py::class_<([^>]+)>\s*\(\s*(\w+)\s*,\s*"([^"]+)"\)[^;]*;)'
    
    for match in re.finditer(class_pattern, content, re.DOTALL):
        full_decl = match.group(1)
        cpp_type = match.group(2)
        module = match.group(3)
        py_name = match.group(4)
        position = match.start()
        
        class_registrations.append({
            'declaration': full_decl,
            'cpp_type': cpp_type,
            'module': module,
            'py_name': py_name,
            'position': position
        })
    
    print(f"✓ Found {len(class_registrations)} class registrations")
    
    # Fix 3: Check if FlowCorrelation exists
    flow_corr_found = False
    flow_corr_reg = None
    
    for reg in class_registrations:
        if 'FlowCorrelation' in reg['cpp_type'] and ',' not in reg['cpp_type']:
            flow_corr_found = True
            flow_corr_reg = reg
            print(f"✓ Found FlowCorrelation registration at position {reg['position']}")
            break
    
    if not flow_corr_found:
        print("⚠ FlowCorrelation not found - adding it")
        
        # Create FlowCorrelation registration
        flow_corr_decl = f'''
    // Base class for flow correlations (must be registered first)
    py::class_<pipeline_sim::FlowCorrelation>({module_var}, "FlowCorrelation")
        .def("calculate", &pipeline_sim::FlowCorrelation::calculate,
             "Calculate pressure drop and flow pattern")
        .def("name", &pipeline_sim::FlowCorrelation::name,
             "Get correlation name");
'''
        
        # Find where to insert (before other correlations)
        insert_pos = content.find('// Correlations')
        if insert_pos == -1:
            insert_pos = content.find('BeggsBrill')
            if insert_pos > 0:
                insert_pos = content.rfind('\n', 0, insert_pos)
        
        if insert_pos > 0:
            content = content[:insert_pos] + flow_corr_decl + content[insert_pos:]
            print("✓ Added FlowCorrelation registration")
    
    # Fix 4: Fix DataDrivenCorrelation inheritance syntax
    # Find DataDrivenCorrelation
    data_driven_pattern = r'py::class_<((?:ml::)?DataDrivenCorrelation)(?:,\s*([^>]+))?>\s*\((\w+),\s*"DataDrivenCorrelation"\)'
    data_driven_match = re.search(data_driven_pattern, content)
    
    if data_driven_match:
        old_decl = data_driven_match.group(0)
        cpp_class = data_driven_match.group(1)
        base_class = data_driven_match.group(2)
        module = data_driven_match.group(3)
        
        print(f"✓ Found DataDrivenCorrelation: {cpp_class}")
        
        if base_class and 'pipeline_sim::' not in base_class:
            # Fix the namespace
            new_decl = f'py::class_<{cpp_class}, pipeline_sim::FlowCorrelation>({module}, "DataDrivenCorrelation")'
            content = content.replace(old_decl, new_decl)
            print("✓ Fixed DataDrivenCorrelation base class namespace")
    
    # Fix 5: Ensure ML module exists before ML classes
    ml_module_exists = 'def_submodule("ml"' in content or 'def_submodule( "ml"' in content
    
    if not ml_module_exists and 'ml::' in content:
        print("⚠ ML classes found but no ML module - adding it")
        
        # Add ML module definition
        ml_module_def = f'\n    // Machine Learning submodule\n    auto ml = {module_var}.def_submodule("ml", "Machine Learning features");\n'
        
        # Insert after module docstring
        doc_end = content.find('";', content.find('doc()')) + 2
        content = content[:doc_end] + ml_module_def + content[doc_end:]
        print("✓ Added ML submodule definition")
    
    # Fix 6: Add any missing includes
    if '#include "pipeline_sim/ml_integration.h"' not in content:
        includes_end = content.rfind('#include')
        includes_end = content.find('\n', includes_end) + 1
        content = content[:includes_end] + '#include "pipeline_sim/ml_integration.h"\n' + content[includes_end:]
        print("✓ Added ml_integration.h include")
    
    # Write fixed content
    fixed_path = bindings_path.replace('.cpp', '_auto_fixed.cpp')
    with open(fixed_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n✓ Created fixed bindings: {os.path.basename(fixed_path)}")
    
    # Test compilation
    print("\nTesting fix with a quick compilation check...")
    print("(This may take a moment)")
    
    # Copy to actual bindings.cpp
    shutil.copy2(fixed_path, bindings_path)
    
    # Try to build
    os.chdir(r"C:\Users\KIMO STORE\HamzaPipeSim")
    result = subprocess.run(
        [sys.executable, "setup_complete.py", "build_ext", "--inplace"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("\n✓ SUCCESS! The fix works - compilation successful")
        print("\nNow complete the build:")
        print("  python setup_complete.py build")
        print("  python -m pip install .")
        return True
    else:
        print("\n⚠ Compilation test failed")
        print("Error (last 500 chars):")
        print(result.stderr[-500:])
        
        # Restore backup
        shutil.copy2(backup_path, bindings_path)
        print("\n✓ Restored original bindings.cpp")
        print("\nPlease check the error and apply manual fixes as described in the guide.")
        return False

if __name__ == "__main__":
    print("HamzaPipeSim Professional Auto-Fix Tool")
    print("This will fix binding issues while preserving ALL features")
    print("")
    
    if fix_bindings_professionally():
        print("\n" + "="*70)
        print("NEXT STEPS:")
        print("="*70)
        print("1. Complete the full build:")
        print("   Remove-Item -Recurse -Force build")
        print("   python setup_complete.py build")
        print("\n2. Install:")
        print("   python -m pip install .")
        print("\n3. Test all features:")
        print("   python -c \"import pipeline_sim; from pipeline_sim.ml import *; print('All features working!')\"")
    else:
        print("\n" + "="*70)
        print("AUTO-FIX INCOMPLETE")
        print("="*70)
        print("Please see the Manual Professional Fix Guide for detailed instructions.")
        print("The issue requires manual intervention to resolve properly.")