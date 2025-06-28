#!/usr/bin/env python3
"""
Fix setup.py Unicode error
"""

import os

print("Fixing setup.py Unicode error")
print("="*60)

# Fix the setup.py file
setup_path = r"C:\Users\KIMO STORE\HamzaPipeSim\setup.py"

# Read the current setup.py
with open(setup_path, 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# Check if it's trying to read README
if 'README' in content and 'fh.read()' in content:
    print("Found README reading code - fixing it")
    
    # Replace the problematic README reading with a simple string
    new_content = content.replace(
        """with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()""",
        """try:
    with open("README.md", "r", encoding="utf-8", errors='ignore') as fh:
        long_description = fh.read()
except:
    long_description = "HamzaPipeSim - Pipeline Network Simulation Package"
"""
    )
    
    # Also handle other variations
    new_content = new_content.replace(
        """with open("README.md", "r") as fh:
    long_description = fh.read()""",
        """try:
    with open("README.md", "r", encoding="utf-8", errors='ignore') as fh:
        long_description = fh.read()
except:
    long_description = "HamzaPipeSim - Pipeline Network Simulation Package"
"""
    )
    
    # Write the fixed content
    with open(setup_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✓ Fixed setup.py")

# Also check if README.md has BOM
readme_path = r"C:\Users\KIMO STORE\HamzaPipeSim\README.md"
if os.path.exists(readme_path):
    # Read and remove BOM if present
    with open(readme_path, 'rb') as f:
        raw = f.read()
    
    # Check for BOM
    if raw.startswith(b'\xff\xfe') or raw.startswith(b'\xfe\xff') or raw.startswith(b'\xef\xbb\xbf'):
        print("Found BOM in README.md - removing it")
        # Remove BOM
        if raw.startswith(b'\xef\xbb\xbf'):
            raw = raw[3:]  # UTF-8 BOM
        elif raw.startswith(b'\xff\xfe'):
            raw = raw[2:]  # UTF-16 LE BOM
        elif raw.startswith(b'\xfe\xff'):
            raw = raw[2:]  # UTF-16 BE BOM
        
        # Write back without BOM
        try:
            text = raw.decode('utf-8', errors='ignore')
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print("✓ Fixed README.md")
        except:
            print("⚠ Could not fix README.md, but setup.py should work anyway")

print("\n✓ Fix complete! Now try installing again:")
print("  python -m pip install .")
print("\nOr use the alternative method:")
print("  python setup_complete.py install")
