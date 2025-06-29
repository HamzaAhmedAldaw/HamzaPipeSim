#!/usr/bin/env python3
"""
Automated Professional Fix for Solver Build Issues
This script will fix all known issues and rebuild the project
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

def print_header(msg):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f" {msg} ")
    print("=" * 70)

def remove_bom_from_file(filepath):
    """Remove BOM from a single file"""
    with open(filepath, 'rb') as f:
        content = f.read()
    
    # Check and remove BOM
    if content.startswith(b'\xef\xbb\xbf'):
        content = content[3:]
        with open(filepath, 'wb') as f:
            f.write(content)
        return True
    return False

def fix_all_bom_issues():
    """Fix BOM in all header and source files"""
    print_header("Fixing BOM Issues")
    
    fixed_count = 0
    for root, dirs, files in os.walk('core'):
        for file in files:
            if file.endswith(('.h', '.hpp', '.cpp', '.c')):
                filepath = os.path.join(root, file)
                if remove_bom_from_file(filepath):
                    print(f"  Fixed BOM in: {filepath}")
                    fixed_count += 1
    
    print(f"\n  Total files fixed: {fixed_count}")
    return fixed_count > 0

def create_clean_solver_cpp():
    """Create a clean solver.cpp file"""
    print_header("Creating Clean solver.cpp")
    
    solver_cpp_content = '''// solver.cpp
#include "pipeline_sim/solver.h"
#include "pipeline_sim/types.h"
#include "pipeline_sim/network.h"
#include "pipeline_sim/node.h"
#include "pipeline_sim/pipe.h"
#include "pipeline_sim/fluid_properties.h"
#include <Eigen/SparseLU>
#include <chrono>
#include <iostream>
#include <cmath>

namespace pipeline_sim {

// Constructor
Solver::Solver(Ptr<Network> network, const FluidProperties& fluid)
    : network_(network), fluid_(fluid), config_() {}

// Destructor
Solver::~Solver() = default;

// Base solve method
SolutionResults Solver::solve() {
    SolutionResults results;
    results.converged = false;
    return results;
}

// SolutionResults methods
Real SolutionResults::pressure_drop(const Ptr<Pipe>& pipe) const {
    auto it = pipe_pressure_drops.find(pipe->id());
    return (it != pipe_pressure_drops.end()) ? it->second : 0.0;
}

Real SolutionResults::outlet_pressure(const Ptr<Pipe>& pipe) const {
    auto it = node_pressures.find(pipe->downstream()->id());
    return (it != node_pressures.end()) ? it->second : 0.0;
}

// SteadyStateSolver implementation
SolutionResults SteadyStateSolver::solve() {
    auto start_time = std::chrono::high_resolution_clock::now();
    SolutionResults results;
    
    size_t num_nodes = network_->nodes().size();
    size_t num_pipes = network_->pipes().size();
    size_t num_unknowns = num_nodes + num_pipes;
    
    Vector x(num_unknowns);
    x.setZero();
    
    for (size_t i = 0; i < num_nodes; ++i) {
        x(i) = constants::STANDARD_PRESSURE;
    }
    
    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        SparseMatrix A(num_unknowns, num_unknowns);
        Vector b(num_unknowns);
        b.setZero();
        
        build_system_matrix(A, b);
        apply_boundary_conditions(A, b);
        
        Eigen::SparseLU<SparseMatrix> solver;
        solver.compute(A);
        
        if (solver.info() != Eigen::Success) {
            std::cerr << "Matrix decomposition failed!" << std::endl;
            break;
        }
        
        Vector dx = solver.solve(b - A * x);
        x = x + config_.relaxation_factor * dx;
        
        Vector residual = A * x - b;
        if (check_convergence(residual)) {
            results.converged = true;
            results.iterations = iter + 1;
            results.residual = residual.norm();
            break;
        }
    }
    
    update_solution(x);
    
    for (const auto& [id, node] : network_->nodes()) {
        results.node_pressures[id] = node->pressure();
        results.node_temperatures[id] = node->temperature();
    }
    
    for (const auto& [id, pipe] : network_->pipes()) {
        results.pipe_flow_rates[id] = pipe->flow_rate();
        results.pipe_pressure_drops[id] = calculate_pressure_drop(pipe);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    results.computation_time = std::chrono::duration<Real>(end_time - start_time).count();
    
    return results;
}

void SteadyStateSolver::build_system_matrix(SparseMatrix& A, Vector& b) {
    std::vector<Eigen::Triplet<Real>> triplets;
    const auto& nodes = network_->nodes();
    const auto& pipes = network_->pipes();
    
    for (const auto& [node_id, node] : nodes) {
        size_t node_idx = network_->node_index(node_id);
        
        if (network_->pressure_specs().count(node_id) > 0) continue;
        
        Real specified_flow = 0.0;
        if (network_->flow_specs().count(node_id) > 0) {
            specified_flow = network_->flow_specs().at(node_id);
        }
        
        for (const auto& pipe : network_->get_upstream_pipes(node)) {
            size_t pipe_idx = network_->pipe_index(pipe->id()) + nodes.size();
            triplets.push_back(Eigen::Triplet<Real>(node_idx, pipe_idx, 1.0));
        }
        
        for (const auto& pipe : network_->get_downstream_pipes(node)) {
            size_t pipe_idx = network_->pipe_index(pipe->id()) + nodes.size();
            triplets.push_back(Eigen::Triplet<Real>(node_idx, pipe_idx, -1.0));
        }
        
        b(node_idx) = specified_flow;
    }
    
    for (const auto& [pipe_id, pipe] : pipes) {
        size_t pipe_idx = network_->pipe_index(pipe_id) + nodes.size();
        size_t upstream_idx = network_->node_index(pipe->upstream()->id());
        size_t downstream_idx = network_->node_index(pipe->downstream()->id());
        
        triplets.push_back(Eigen::Triplet<Real>(pipe_idx, upstream_idx, 1.0));
        triplets.push_back(Eigen::Triplet<Real>(pipe_idx, downstream_idx, -1.0));
        
        Real density = fluid_.mixture_density();
        Real viscosity = fluid_.mixture_viscosity();
        Real area = pipe->area();
        Real q = pipe->flow_rate();
        if (std::abs(q) < 1e-6) q = 0.001;
        
        Real velocity = q / area;
        Real reynolds = pipe->reynolds_number(viscosity, density);
        Real friction = pipe->friction_factor(reynolds);
        Real resistance = friction * pipe->length() * density * std::abs(velocity) / 
                         (2.0 * pipe->diameter() * area);
        
        triplets.push_back(Eigen::Triplet<Real>(pipe_idx, pipe_idx, -resistance));
        
        Real dz = pipe->downstream()->elevation() - pipe->upstream()->elevation();
        b(pipe_idx) = -density * constants::GRAVITY * dz;
    }
    
    A.setFromTriplets(triplets.begin(), triplets.end());
}

void SteadyStateSolver::apply_boundary_conditions(SparseMatrix& A, Vector& b) {
    for (const auto& [node_id, pressure] : network_->pressure_specs()) {
        size_t idx = network_->node_index(node_id);
        
        for (int k = 0; k < A.outerSize(); ++k) {
            for (SparseMatrix::InnerIterator it(A, k); it; ++it) {
                if (it.row() == idx) {
                    it.valueRef() = (it.col() == idx) ? 1.0 : 0.0;
                }
            }
        }
        
        b(idx) = pressure;
    }
}

void SteadyStateSolver::update_solution(const Vector& x) {
    const auto& nodes = network_->nodes();
    const auto& pipes = network_->pipes();
    
    for (const auto& [node_id, node] : nodes) {
        size_t idx = network_->node_index(node_id);
        node->set_pressure(x(idx));
    }
    
    for (const auto& [pipe_id, pipe] : pipes) {
        size_t idx = network_->pipe_index(pipe_id) + nodes.size();
        pipe->set_flow_rate(x(idx));
    }
}

bool SteadyStateSolver::check_convergence(const Vector& residual) {
    return residual.norm() < config_.tolerance;
}

Real SteadyStateSolver::calculate_pressure_drop(const Ptr<Pipe>& pipe) {
    Real density = fluid_.mixture_density();
    Real viscosity = fluid_.mixture_viscosity();
    Real velocity = pipe->velocity();
    Real reynolds = pipe->reynolds_number(viscosity, density);
    Real friction = pipe->friction_factor(reynolds);
    
    Real friction_dp = friction * pipe->length() * density * velocity * velocity / 
                      (2.0 * pipe->diameter());
    Real dz = pipe->downstream()->elevation() - pipe->upstream()->elevation();
    Real gravity_dp = density * constants::GRAVITY * dz;
    
    return friction_dp + gravity_dp;
}

} // namespace pipeline_sim
'''
    
    # Backup existing file
    solver_path = Path('core/src/solver.cpp')
    if solver_path.exists():
        backup_path = solver_path.with_suffix('.cpp.backup')
        shutil.copy2(solver_path, backup_path)
        print(f"  Backed up existing file to: {backup_path}")
    
    # Write new file
    with open(solver_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(solver_cpp_content)
    
    print(f"  Created clean solver.cpp at: {solver_path}")

def clean_build_directory():
    """Clean build artifacts"""
    print_header("Cleaning Build Directory")
    
    dirs_to_clean = ['build', 'dist', '*.egg-info']
    
    for pattern in dirs_to_clean:
        if '*' in pattern:
            # Handle glob patterns
            from glob import glob
            for path in glob(pattern):
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                    print(f"  Removed: {path}")
        else:
            # Handle regular paths
            if os.path.exists(pattern):
                shutil.rmtree(pattern, ignore_errors=True)
                print(f"  Removed: {pattern}")

def rebuild_project():
    """Rebuild the project"""
    print_header("Rebuilding Project")
    
    cmd = [sys.executable, 'setup_complete.py', 'build_ext', '--inplace']
    print(f"  Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("\n  ✓ Build successful!")
    else:
        print("\n  ✗ Build failed!")
        print("\nSTDOUT:", result.stdout[-1000:])  # Last 1000 chars
        print("\nSTDERR:", result.stderr[-1000:])

def main():
    """Main fix routine"""
    print("=" * 70)
    print(" HamzaPipeSim Professional Build Fix ")
    print("=" * 70)
    
    # 1. Fix BOM issues
    fix_all_bom_issues()
    
    # 2. Create clean solver.cpp
    create_clean_solver_cpp()
    
    # 3. Clean build directory
    clean_build_directory()
    
    # 4. Rebuild
    rebuild_project()
    
    print_header("COMPLETE")
    print("If the build still fails, please check:")
    print("1. All vcpkg dependencies are installed")
    print("2. MSVC compiler is properly configured")
    print("3. Python environment has all required packages")

if __name__ == "__main__":
    main()