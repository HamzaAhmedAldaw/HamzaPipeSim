"""
Fix UTF-8 BOM issues in header files that might be causing compilation problems
"""

import os
import glob

def remove_bom(file_path):
    """Remove UTF-8 BOM from file if present"""
    with open(file_path, 'rb') as f:
        content = f.read()
    
    # Check for UTF-8 BOM (EF BB BF)
    if content.startswith(b'\xef\xbb\xbf'):
        print(f"Found BOM in {file_path}, removing...")
        # Remove BOM and write back
        with open(file_path, 'wb') as f:
            f.write(content[3:])
        return True
    return False

def fix_all_headers():
    """Fix all header files in the project"""
    header_patterns = [
        'core/include/pipeline_sim/*.h',
        'core/include/pipeline_sim/*.hpp',
    ]
    
    fixed_count = 0
    
    for pattern in header_patterns:
        for file_path in glob.glob(pattern, recursive=True):
            if remove_bom(file_path):
                fixed_count += 1
    
    print(f"\nFixed {fixed_count} files with BOM issues")
    
    # Specifically check solver.h
    solver_h = 'core/include/pipeline_sim/solver.h'
    if os.path.exists(solver_h):
        print(f"\nChecking {solver_h}...")
        with open(solver_h, 'rb') as f:
            first_bytes = f.read(10)
            print(f"First bytes: {first_bytes}")

if __name__ == "__main__":
    print("Fixing UTF-8 BOM issues in header files...")
    fix_all_headers()
    print("\nDone! Try building again.")