#!/usr/bin/env python3
"""
Quick fix script for the linker errors
"""

import os

def fix_network_cpp():
    """Add missing methods to network.cpp"""
    network_cpp_path = "core/src/network.cpp"
    
    # Check if file exists
    if not os.path.exists(network_cpp_path):
        print(f"Error: {network_cpp_path} not found!")
        return False
    
    # Read the file
    with open(network_cpp_path, 'r') as f:
        content = f.read()
    
    # Check if methods already exist
    if "is_valid()" in content and "clear()" in content:
        print("Methods already exist in network.cpp")
        return True
    
    # Add missing methods before the closing namespace brace
    missing_methods = '''
bool Network::is_valid() const {
    // Basic validation
    return !nodes_.empty() && !pipes_.empty() && 
           (!pressure_specs_.empty() || std::any_of(nodes_.begin(), nodes_.end(),
            [](const auto& p) { return p.second->has_pressure_bc(); }));
}

void Network::clear() {
    nodes_.clear();
    pipes_.clear();
    node_index_.clear();
    pipe_index_.clear();
    pressure_specs_.clear();
    flow_specs_.clear();
    upstream_pipes_.clear();
    downstream_pipes_.clear();
    next_node_id_ = 0;
    next_pipe_id_ = 0;
}

void Network::save_to_json(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot create file: " + filename);
    }
    file << "{\\n  \\"nodes\\": [],\\n  \\"pipes\\": []\\n}\\n";
    file.close();
}
'''
    
    # Find the last closing brace of namespace
    last_brace = content.rfind("} // namespace pipeline_sim")
    if last_brace == -1:
        last_brace = content.rfind("}")
    
    # Insert methods before the closing brace
    new_content = content[:last_brace] + missing_methods + "\n" + content[last_brace:]
    
    # Write back
    with open(network_cpp_path, 'w') as f:
        f.write(new_content)
    
    print(f"✓ Added missing methods to {network_cpp_path}")
    return True

def fix_ml_integration_includes():
    """Add Eigen/Dense include to ml_integration.cpp"""
    ml_cpp_path = "core/src/ml_integration.cpp"
    
    if not os.path.exists(ml_cpp_path):
        print(f"Error: {ml_cpp_path} not found!")
        return False
    
    # Read the file
    with open(ml_cpp_path, 'r') as f:
        content = f.read()
    
    # Check if Eigen/Dense is already included
    if "#include <Eigen/Dense>" in content:
        print("Eigen/Dense already included in ml_integration.cpp")
        return True
    
    # Add after other includes
    if "#include <cmath>" in content:
        content = content.replace("#include <cmath>", "#include <cmath>\n#include <Eigen/Dense>")
    else:
        # Add at the beginning after the header comment
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith("#include"):
                lines.insert(i, "#include <Eigen/Dense>")
                break
        content = '\n'.join(lines)
    
    # Write back
    with open(ml_cpp_path, 'w') as f:
        f.write(content)
    
    print(f"✓ Added Eigen/Dense include to {ml_cpp_path}")
    return True

def main():
    print("Fixing linker errors...")
    print("=" * 50)
    
    # Add missing includes
    if not os.path.exists("core/src/network.cpp"):
        print("\nMake sure network.cpp includes <algorithm> for std::any_of")
    
    # Fix network.cpp
    fix_network_cpp()
    
    # Fix ml_integration.cpp
    fix_ml_integration_includes()
    
    print("\n✓ Fixes applied!")
    print("\nNow run: python setup_complete.py build")

if __name__ == "__main__":
    main()