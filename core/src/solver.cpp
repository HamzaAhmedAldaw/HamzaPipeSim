// ===== solver.cpp =====
#include "pipeline_sim/solver.h"
#include <Eigen/SparseLU>
#include <chrono>
#include <iostream>

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
    
    // Get problem dimensions
    size_t num_nodes = network_->nodes().size();
    size_t num_pipes = network_->pipes().size();
    size_t num_unknowns = num_nodes + num_pipes;
    
    // Initialize solution vectors
    Vector x(num_unknowns);
    Vector x_old(num_unknowns);
    x.setZero();
    
    // Initial guess - atmospheric pressure and zero flow
    for (size_t i = 0; i < num_nodes; ++i) {
        x(i) = constants::STANDARD_PRESSURE;
    }
    
    // Main iteration loop
    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        x_old = x;
        
        // Build system matrix
        SparseMatrix A(num_unknowns, num_unknowns);
        Vector b(num_unknowns);
        b.setZero();
        
        build_system_matrix(A, b);
        apply_boundary_conditions(A, b);
        
        // Solve linear system
        Eigen::SparseLU<SparseMatrix> solver;
        solver.compute(A);
        
        if (solver.info() != Eigen::Success) {
            std::cerr << "Matrix decomposition failed!" << std::endl;
            break;
        }
        
        Vector dx = solver.solve(b - A * x);
        
        // Apply relaxation
        x = x + config_.relaxation_factor * dx;
        
        // Check convergence
        Vector residual = A * x - b;
        if (check_convergence(residual)) {
            results.converged = true;
            results.iterations = iter + 1;
            results.residual = residual.norm();
            break;
        }
        
        if (config_.verbose && iter % 10 == 0) {
            std::cout << "Iteration " << iter << ": residual = " 
                     << residual.norm() << std::endl;
        }
    }
    
    // Extract solution
    update_solution(x);
    
    // Store results
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
            triplets.push_back(Eigen::Triplet<Real>(node_idx, pipe_idx, 1.0));
        }
        
        // Downstream pipes contribute negative flow
        for (const auto& pipe : network_->get_downstream_pipes(node)) {
            size_t pipe_idx = network_->pipe_index(pipe->id()) + nodes.size();
            triplets.push_back(Eigen::Triplet<Real>(node_idx, pipe_idx, -1.0));
        }
        
        b(node_idx) = specified_flow;
    }
    
    // Momentum equations for pipes
    for (const auto& [pipe_id, pipe] : pipes) {
        size_t pipe_idx = network_->pipe_index(pipe_id) + nodes.size();
        size_t upstream_idx = network_->node_index(pipe->upstream()->id());
        size_t downstream_idx = network_->node_index(pipe->downstream()->id());
        
        // Pressure difference drives flow
        triplets.push_back(Eigen::Triplet<Real>(pipe_idx, upstream_idx, 1.0));
        triplets.push_back(Eigen::Triplet<Real>(pipe_idx, downstream_idx, -1.0));
        
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
        
        triplets.push_back(Eigen::Triplet<Real>(pipe_idx, pipe_idx, -resistance));
        
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
        
        // Set row to identity
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