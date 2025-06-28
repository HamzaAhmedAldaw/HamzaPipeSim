// ===== solver.cpp =====
#include "pipeline_sim/solver.h"
#include <Eigen/SparseLU>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>

namespace pipeline_sim {

Solver::Solver(Ptr<Network> network, const FluidProperties& fluid)
    : network_(network), fluid_(fluid) {
}

SolutionResults Solver::solve() {
    SolutionResults results;
    results.converged = false;
    return results;
}

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
    results.converged = false;
    results.iterations = 0;
    results.residual = 1e10;
    results.max_mass_imbalance = 0.0;
    
    if (config_.verbose) {
        std::cout << "=== Starting Commercial-Grade Steady-State Solver ===" << std::endl;
        std::cout << "Network: " << network_->nodes().size() << " nodes, " 
                  << network_->pipes().size() << " pipes" << std::endl;
        std::cout << "Pressure BCs: " << network_->pressure_specs().size() 
                  << ", Flow BCs: " << network_->flow_specs().size() << std::endl;
        std::cout << "Fluid: density = " << fluid_.mixture_density() 
                  << " kg/m³, viscosity = " << fluid_.mixture_viscosity() * 1000 << " cP" << std::endl;
    }
    
    // Get problem dimensions
    size_t num_nodes = network_->nodes().size();
    size_t num_pipes = network_->pipes().size();
    size_t num_unknowns = num_nodes + num_pipes;
    
    // Initialize solution vectors
    Vector x(num_unknowns);
    Vector x_old(num_unknowns);
    x.setZero();
    
    // Initial guess - use specified pressures where available
    for (size_t i = 0; i < num_nodes; ++i) {
        x(i) = constants::STANDARD_PRESSURE;
    }
    
    // Set initial pressures from boundary conditions
    for (const auto& [node_id, pressure] : network_->pressure_specs()) {
        size_t idx = network_->node_index(node_id);
        x(idx) = pressure;
    }
    
    // Main iteration loop
    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        x_old = x;
        
        // Build system matrix
        SparseMatrix A(num_unknowns, num_unknowns);
        Vector b(num_unknowns);
        b.setZero();
        
        // Reserve space for triplets (optimization)
        std::vector<Eigen::Triplet<Real>> triplets;
        triplets.reserve(num_nodes * 10 + num_pipes * 5);
        
        build_system_matrix(A, b, triplets);
        apply_boundary_conditions(A, b);
        
        // Solve linear system
        Eigen::SparseLU<SparseMatrix> solver;
        solver.analyzePattern(A);
        solver.factorize(A);
        
        if (solver.info() != Eigen::Success) {
            if (config_.verbose) {
                std::cerr << "Matrix decomposition failed at iteration " << iter << std::endl;
            }
            break;
        }
        
        Vector dx = solver.solve(b - A * x);
        
        if (solver.info() != Eigen::Success) {
            if (config_.verbose) {
                std::cerr << "Linear solve failed at iteration " << iter << std::endl;
            }
            break;
        }
        
        // Apply relaxation
        x = x + config_.relaxation_factor * dx;
        
        // Update solution
        update_solution(x);
        
        // Check convergence
        Vector residual = A * x - b;
        Real residual_norm = residual.norm();
        
        if (config_.verbose && iter % 10 == 0) {
            std::cout << "Iteration " << iter << ": residual = " 
                     << residual_norm << std::endl;
        }
        
        if (residual_norm < config_.tolerance) {
            results.converged = true;
            results.iterations = iter + 1;
            results.residual = residual_norm;
            
            if (config_.verbose) {
                std::cout << "Converged in " << results.iterations 
                         << " iterations, residual = " << residual_norm << std::endl;
            }
            break;
        }
        
        // Check for divergence
        if (residual_norm > 1e10 || std::isnan(residual_norm)) {
            if (config_.verbose) {
                std::cerr << "Solution diverged at iteration " << iter << std::endl;
            }
            break;
        }
    }
    
    // Calculate mass imbalances
    Real max_imbalance = 0.0;
    for (const auto& [node_id, node] : network_->nodes()) {
        if (network_->pressure_specs().count(node_id) > 0) continue;
        
        Real imbalance = 0.0;
        // Sum flows in
        for (const auto& pipe : network_->get_upstream_pipes(node)) {
            imbalance += pipe->flow_rate() * fluid_.mixture_density();
        }
        // Sum flows out  
        for (const auto& pipe : network_->get_downstream_pipes(node)) {
            imbalance -= pipe->flow_rate() * fluid_.mixture_density();
        }
        
        max_imbalance = std::max(max_imbalance, std::abs(imbalance));
    }
    results.max_mass_imbalance = max_imbalance;
    
    // Store results
    for (const auto& [id, node] : network_->nodes()) {
        results.node_pressures[id] = node->pressure();
        results.node_temperatures[id] = node->temperature();
    }
    
    for (const auto& [id, pipe] : network_->pipes()) {
        results.pipe_flow_rates[id] = pipe->flow_rate();
        results.pipe_pressure_drops[id] = calculate_pressure_drop(pipe);
        results.pipe_velocities[id] = pipe->velocity();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    results.computation_time = std::chrono::duration<Real>(end_time - start_time).count();
    
    return results;
}

void SteadyStateSolver::build_system_matrix(SparseMatrix& A, Vector& b) {
    std::vector<Eigen::Triplet<Real>> triplets;
    build_system_matrix(A, b, triplets);
}

void SteadyStateSolver::build_system_matrix(SparseMatrix& A, Vector& b, 
                                           std::vector<Eigen::Triplet<Real>>& triplets) {
    // Node indices
    const auto& nodes = network_->nodes();
    const auto& pipes = network_->pipes();
    
    // Mass conservation equations at nodes
    for (const auto& [node_id, node] : nodes) {
        size_t node_idx = network_->node_index(node_id);
        
        // Skip if pressure is specified
        if (network_->pressure_specs().count(node_id) > 0) continue;
        
        // Sum of flows = 0 (or specified flow)
        Real specified_flow = 0.0;
        if (network_->flow_specs().count(node_id) > 0) {
            specified_flow = network_->flow_specs().at(node_id);
        }
        
        // Upstream pipes contribute positive flow
        for (const auto& pipe : network_->get_upstream_pipes(node)) {
            size_t pipe_idx = network_->pipe_index(pipe->id()) + nodes.size();
            triplets.push_back(Eigen::Triplet<Real>(static_cast<int>(node_idx), static_cast<int>(pipe_idx), 1.0));
        }
        
        // Downstream pipes contribute negative flow
        for (const auto& pipe : network_->get_downstream_pipes(node)) {
            size_t pipe_idx = network_->pipe_index(pipe->id()) + nodes.size();
            triplets.push_back(Eigen::Triplet<Real>(static_cast<int>(node_idx), static_cast<int>(pipe_idx), -1.0));
        }
        
        b(node_idx) = specified_flow;
    }
    
    // Momentum equations for pipes
    for (const auto& [pipe_id, pipe] : pipes) {
        size_t pipe_idx = network_->pipe_index(pipe_id) + nodes.size();
        size_t upstream_idx = network_->node_index(pipe->upstream()->id());
        size_t downstream_idx = network_->node_index(pipe->downstream()->id());
        
        // Pressure difference drives flow
        triplets.push_back(Eigen::Triplet<Real>(static_cast<int>(pipe_idx), static_cast<int>(upstream_idx), 1.0));
        triplets.push_back(Eigen::Triplet<Real>(static_cast<int>(pipe_idx), static_cast<int>(downstream_idx), -1.0));
        
        // Flow resistance (linearized)
        Real density = fluid_.mixture_density();
        Real viscosity = fluid_.mixture_viscosity();
        Real area = pipe->area();
        
        // Assume initial flow for linearization
        Real q = pipe->flow_rate();
        if (std::abs(q) < 1e-6) q = 0.001;  // Avoid division by zero
        
        Real velocity = q / area;
        Real reynolds = pipe->reynolds_number(viscosity, density);
        Real friction = pipe->friction_factor(reynolds);
        
        // Resistance coefficient
        Real resistance = friction * pipe->length() * density * std::abs(velocity) / 
                         (2.0 * pipe->diameter() * area);
        
        triplets.push_back(Eigen::Triplet<Real>(static_cast<int>(pipe_idx), static_cast<int>(pipe_idx), -resistance));
        
        // Gravitational term
        Real dz = pipe->downstream()->elevation() - pipe->upstream()->elevation();
        b(pipe_idx) = -density * constants::GRAVITY * dz;
    }
    
    A.setFromTriplets(triplets.begin(), triplets.end());
}

void SteadyStateSolver::apply_boundary_conditions(SparseMatrix& A, Vector& b) {
    const auto& nodes = network_->nodes();
    
    // Apply pressure boundary conditions
    for (const auto& [node_id, pressure] : network_->pressure_specs()) {
        size_t idx = network_->node_index(node_id);
        
        // Method 1: Direct modification after matrix assembly
        // This is more efficient for sparse matrices
        
        // Clear the row
        for (int k = 0; k < A.outerSize(); ++k) {
            for (SparseMatrix::InnerIterator it(A, k); it; ++it) {
                if (it.row() == idx) {
                    it.valueRef() = 0.0;
                }
            }
        }
        
        // Set diagonal to 1
        A.coeffRef(static_cast<int>(idx), static_cast<int>(idx)) = 1.0;
        
        // Update RHS
        b(idx) = pressure;
        
        // Also need to remove this equation's influence on other equations
        // This is done by zeroing the column (except diagonal)
        for (int i = 0; i < A.rows(); ++i) {
            if (i != idx) {
                A.coeffRef(i, static_cast<int>(idx)) = 0.0;
            }
        }
    }
    
    // Make sure matrix is compressed after modifications
    A.makeCompressed();
}

void SteadyStateSolver::update_solution(const Vector& x) {
    const auto& nodes = network_->nodes();
    const auto& pipes = network_->pipes();
    
    // Update node pressures
    for (const auto& [node_id, node] : nodes) {
        size_t idx = network_->node_index(node_id);
        node->set_pressure(x(idx));
    }
    
    // Update pipe flows
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
    
    // Darcy-Weisbach equation
    Real friction_dp = friction * pipe->length() * density * velocity * velocity / 
                      (2.0 * pipe->diameter());
    
    // Gravitational pressure drop
    Real dz = pipe->downstream()->elevation() - pipe->upstream()->elevation();
    Real gravity_dp = density * constants::GRAVITY * dz;
    
    return friction_dp + gravity_dp;
}

} // namespace pipeline_sim
