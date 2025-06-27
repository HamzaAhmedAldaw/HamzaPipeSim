#include "pipeline_sim/solver.h"
#include "pipeline_sim/correlations.h"
#include <iostream>
#include <iomanip>
#include <chrono>

namespace pipeline_sim {

// Solution Results methods
Real SolutionResults::pressure_drop(const Ptr<Pipe>& pipe) const {
    auto it = pipe_pressure_drops.find(pipe->id());
    return it != pipe_pressure_drops.end() ? it->second : 0.0;
}

Real SolutionResults::outlet_pressure(const Ptr<Pipe>& pipe) const {
    auto it = node_pressures.find(pipe->downstream()->id());
    return it != node_pressures.end() ? it->second : 0.0;
}

// Base Solver
Solver::Solver(Ptr<Network> network, const FluidProperties& fluid)
    : network_(network), fluid_(fluid) {
    config_.tolerance = 1e-6;
    config_.max_iterations = 100;
    config_.relaxation_factor = 1.0;
    config_.verbose = false;
}

SolutionResults Solver::solve() {
    // Base implementation - derived classes override
    return SolutionResults();
}

// Steady State Solver
SolutionResults SteadyStateSolver::solve() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    SolutionResults results;
    
    // Determine system size
    size_t n_nodes = network_->nodes().size();
    size_t n_pipes = network_->pipes().size();
    size_t n = n_nodes + n_pipes;
    
    // Solution vector: [node pressures, pipe flow rates]
    Vector x(static_cast<int>(n));
    Vector x_old(static_cast<int>(n));
    
    // Initialize with reasonable guesses
    size_t idx = 0;
    for (const auto& [id, node] : network_->nodes()) {
        x(static_cast<int>(idx++)) = node->pressure();
    }
    for (const auto& [id, pipe] : network_->pipes()) {
        x(static_cast<int>(idx++)) = pipe->flow_rate();
    }
    
    // Newton-Raphson iteration
    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        x_old = x;
        
        // Build system matrix and RHS
        SparseMatrix A(static_cast<int>(n), static_cast<int>(n));
        Vector b(static_cast<int>(n));
        
        build_system_matrix(A, b);
        
        // Apply boundary conditions
        apply_boundary_conditions(A, b);
        
        // Solve linear system
        Eigen::SparseLU<SparseMatrix> solver;
        solver.compute(A);
        
        if (solver.info() != Eigen::Success) {
            std::cerr << "Matrix decomposition failed!" << std::endl;
            results.converged = false;
            results.iterations = iter;
            return results;
        }
        
        Vector dx = solver.solve(b);
        
        if (solver.info() != Eigen::Success) {
            std::cerr << "Solution failed!" << std::endl;
            results.converged = false;
            results.iterations = iter;
            return results;
        }
        
        // Update solution with relaxation
        x = x_old + config_.relaxation_factor * dx;
        
        // Update network state
        update_solution(x);
        
        // Check convergence
        Vector residual = A * x - b;
        results.residual = residual.norm();
        
        if (config_.verbose && iter % 10 == 0) {
            std::cout << "Iteration " << iter << ": residual = " 
                     << std::scientific << results.residual << std::endl;
        }
        
        if (check_convergence(residual)) {
            results.converged = true;
            results.iterations = iter + 1;
            break;
        }
    }
    
    // Store final results
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
    
    if (config_.verbose) {
        std::cout << "\nSolution completed in " << results.iterations 
                 << " iterations, " << results.computation_time << " seconds" << std::endl;
    }
    
    return results;
}

void SteadyStateSolver::build_system_matrix(SparseMatrix& A, Vector& b) {
    std::vector<Eigen::Triplet<Real>> triplets;
    
    size_t n_nodes = network_->nodes().size();
    size_t n_pipes = network_->pipes().size();
    
    // Node equations (mass conservation)
    size_t node_idx = 0;
    for (const auto& [node_id, node] : network_->nodes()) {
        // Mass conservation: sum of flows = 0
        for (const auto& pipe : network_->get_upstream_pipes(node)) {
            auto pipe_it = network_->pipes().find(pipe->id());
            if (pipe_it != network_->pipes().end()) {
                size_t pipe_idx = n_nodes + std::distance(network_->pipes().begin(), pipe_it);
                triplets.push_back(Eigen::Triplet<Real>(
                    static_cast<int>(node_idx), 
                    static_cast<int>(pipe_idx), 
                    1.0));
            }
        }
        
        for (const auto& pipe : network_->get_downstream_pipes(node)) {
            auto pipe_it = network_->pipes().find(pipe->id());
            if (pipe_it != network_->pipes().end()) {
                size_t pipe_idx = n_nodes + std::distance(network_->pipes().begin(), pipe_it);
                triplets.push_back(Eigen::Triplet<Real>(
                    static_cast<int>(node_idx), 
                    static_cast<int>(pipe_idx), 
                    -1.0));
            }
        }
        
        // Source/sink terms
        if (node->type() == NodeType::SOURCE || node->type() == NodeType::SINK) {
            b(static_cast<int>(node_idx)) = node->fixed_flow_rate();
        } else {
            b(static_cast<int>(node_idx)) = 0.0;
        }
        
        node_idx++;
    }
    
    // Pipe momentum equations
    size_t pipe_idx = n_nodes;
    for (const auto& [pipe_id, pipe] : network_->pipes()) {
        auto up_node_it = network_->nodes().find(pipe->upstream()->id());
        auto down_node_it = network_->nodes().find(pipe->downstream()->id());
        
        if (up_node_it != network_->nodes().end() && down_node_it != network_->nodes().end()) {
            size_t up_idx = std::distance(network_->nodes().begin(), up_node_it);
            size_t down_idx = std::distance(network_->nodes().begin(), down_node_it);
            
            // Momentum equation: P_up - P_down - ?P_friction - ?P_elevation = 0
            triplets.push_back(Eigen::Triplet<Real>(
                static_cast<int>(pipe_idx), 
                static_cast<int>(up_idx), 
                1.0));
            triplets.push_back(Eigen::Triplet<Real>(
                static_cast<int>(pipe_idx), 
                static_cast<int>(down_idx), 
                -1.0));
            
            // Friction term (linearized)
            Real dp_friction = calculate_pressure_drop(pipe);
            Real flow = pipe->flow_rate();
            
            if (std::abs(flow) > 1e-10) {
                // d?P/dQ ˜ 2*?P/Q for turbulent flow
                Real friction_derivative = 2.0 * dp_friction / flow;
                triplets.push_back(Eigen::Triplet<Real>(
                    static_cast<int>(pipe_idx), 
                    static_cast<int>(pipe_idx), 
                    -friction_derivative));
            }
            
            // RHS: elevation pressure drop
            Real dp_elevation = fluid_.mixture_density() * constants::GRAVITY * 
                               (pipe->downstream()->elevation() - pipe->upstream()->elevation());
            b(static_cast<int>(pipe_idx)) = -dp_elevation - dp_friction;
        }
        
        pipe_idx++;
    }
    
    A.setFromTriplets(triplets.begin(), triplets.end());
}

void SteadyStateSolver::apply_boundary_conditions(SparseMatrix& A, Vector& b) {
    size_t idx = 0;
    
    // Apply pressure boundary conditions
    for (const auto& [id, node] : network_->nodes()) {
        if (node->has_pressure_bc()) {
            // Replace equation with P = P_bc
            for (int k = 0; k < A.outerSize(); ++k) {
                for (SparseMatrix::InnerIterator it(A, k); it; ++it) {
                    if (it.row() == static_cast<int>(idx) || it.col() == static_cast<int>(idx)) {
                        it.valueRef() = (it.row() == it.col()) ? 1.0 : 0.0;
                    }
                }
            }
            b(static_cast<int>(idx)) = node->pressure_bc();
        }
        idx++;
    }
    
    // Apply flow boundary conditions
    size_t n_nodes = network_->nodes().size();
    idx = 0;
    for (const auto& [id, pipe] : network_->pipes()) {
        if (pipe->has_flow_bc()) {
            size_t pipe_idx = n_nodes + idx;
            for (int k = 0; k < A.outerSize(); ++k) {
                for (SparseMatrix::InnerIterator it(A, k); it; ++it) {
                    if (it.row() == static_cast<int>(pipe_idx) || it.col() == static_cast<int>(pipe_idx)) {
                        it.valueRef() = (it.row() == it.col()) ? 1.0 : 0.0;
                    }
                }
            }
            b(static_cast<int>(pipe_idx)) = pipe->flow_bc();
        }
        idx++;
    }
}

void SteadyStateSolver::update_solution(const Vector& x) {
    size_t idx = 0;
    
    // Update node pressures
    for (const auto& [id, node] : network_->nodes()) {
        node->set_pressure(x(static_cast<int>(idx++)));
    }
    
    // Update pipe flow rates
    for (const auto& [id, pipe] : network_->pipes()) {
        pipe->set_flow_rate(x(static_cast<int>(idx++)));
        
        // Update velocities
        Real velocity = pipe->flow_rate() / pipe->area();
        pipe->set_velocity(velocity);
    }
}

bool SteadyStateSolver::check_convergence(const Vector& residual) {
    return residual.norm() < config_.tolerance;
}

Real SteadyStateSolver::calculate_pressure_drop(const Ptr<Pipe>& pipe) {
    // Simple pressure drop calculation
    Real flow_rate = pipe->flow_rate();
    Real velocity = flow_rate / pipe->area();
    Real density = fluid_.mixture_density();
    Real viscosity = fluid_.mixture_viscosity();
    
    // Reynolds number
    Real reynolds = density * std::abs(velocity) * pipe->diameter() / viscosity;
    
    // Friction factor
    Real f = pipe->friction_factor(reynolds);
    
    // Darcy-Weisbach equation
    Real dp_friction = f * (pipe->length() / pipe->diameter()) * 
                      (0.5 * density * velocity * velocity);
    
    // Elevation pressure drop
    Real dp_elevation = density * constants::GRAVITY * 
                       (pipe->downstream()->elevation() - pipe->upstream()->elevation());
    
    return dp_friction + dp_elevation;
}

} // namespace pipeline_sim
