// ===== solver.cpp - Professional Implementation =====
#include "pipeline_sim/solver.h"
#include <Eigen/SparseLU>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <exception>

namespace pipeline_sim {

// ===== Base Solver Implementation =====
Solver::Solver(Ptr<Network> network, const FluidProperties& fluid)
    : network_(network), fluid_(fluid) {
    // Initialize default configuration
    config_.tolerance = 1e-6;
    config_.max_iterations = 100;
    config_.relaxation_factor = 1.0;
    config_.verbose = false;
    config_.min_flow_velocity = 1e-6;
    config_.pressure_damping = 0.7;
}

SolutionResults Solver::solve() {
    // Base class returns empty non-converged results
    SolutionResults results;
    results.converged = false;
    results.iterations = 0;
    results.residual = 1e10;
    results.computation_time = 0.0;
    return results;
}

// ===== SolutionResults Implementation =====
Real SolutionResults::pressure_drop(const Ptr<Pipe>& pipe) const {
    auto it = pipe_pressure_drops.find(pipe->id());
    return (it != pipe_pressure_drops.end()) ? it->second : 0.0;
}

Real SolutionResults::outlet_pressure(const Ptr<Pipe>& pipe) const {
    auto it = node_pressures.find(pipe->downstream()->id());
    return (it != node_pressures.end()) ? it->second : 0.0;
}

// ===== SteadyStateSolver Implementation =====
SolutionResults SteadyStateSolver::solve() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Initialize results with default values
    SolutionResults results;
    results.converged = false;
    results.iterations = 0;
    results.residual = 1e10;
    
    // Validate network
    if (!network_ || network_->nodes().empty()) {
        std::cerr << "ERROR: Invalid network" << std::endl;
        return results;
    }
    
    if (config_.verbose) {
        std::cout << "\n=== Starting Steady-State Solver ===" << std::endl;
        std::cout << "Network: " << network_->nodes().size() << " nodes, " 
                  << network_->pipes().size() << " pipes" << std::endl;
        std::cout << "Fluid: density = " << fluid_.mixture_density() 
                  << " kg/m³, viscosity = " << fluid_.mixture_viscosity() * 1000 << " cP" << std::endl;
    }
    
    // Solve using pressure-only formulation
    bool converged = false;
    int iterations_performed = 0;
    Real final_residual = 1e10;
    
    try {
        // Use pressure-based Newton-Raphson method
        converged = solve_pressure_based(iterations_performed, final_residual);
    } catch (const std::exception& e) {
        std::cerr << "ERROR in solver: " << e.what() << std::endl;
    }
    
    // Update results
    results.converged = converged;
    results.iterations = iterations_performed;
    results.residual = final_residual;
    
    // Extract solution regardless of convergence
    extract_results(results);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    results.computation_time = std::chrono::duration<Real>(end_time - start_time).count();
    
    if (config_.verbose) {
        std::cout << "\nSolver completed:" << std::endl;
        std::cout << "  Converged: " << (converged ? "YES" : "NO") << std::endl;
        std::cout << "  Iterations: " << iterations_performed << std::endl;
        std::cout << "  Final residual: " << std::scientific << final_residual << std::endl;
        std::cout << "  Time: " << std::fixed << std::setprecision(3) 
                  << results.computation_time << " seconds" << std::endl;
    }
    
    return results;
}

bool SteadyStateSolver::solve_pressure_based(int& iterations_performed, Real& final_residual) {
    const auto& nodes = network_->nodes();
    const auto& pipes = network_->pipes();
    
    // Create node ordering
    std::vector<std::string> node_ids;
    for (const auto& [id, node] : nodes) {
        node_ids.push_back(id);
    }
    
    size_t n = node_ids.size();
    
    // Initialize pressure vector
    Vector p(n);
    for (size_t i = 0; i < n; ++i) {
        const auto& node = nodes.at(node_ids[i]);
        p(i) = node->pressure();
        
        // Use boundary condition if available
        if (network_->pressure_specs().count(node_ids[i]) > 0) {
            p(i) = network_->pressure_specs().at(node_ids[i]);
        }
    }
    
    // Newton-Raphson iteration
    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        iterations_performed = iter + 1;
        
        // Build Jacobian and residual
        SparseMatrix J(n, n);
        Vector F(n);
        
        if (!build_jacobian_and_residual(J, F, p, node_ids)) {
            std::cerr << "ERROR: Failed to build system" << std::endl;
            return false;
        }
        
        // Compute residual norm
        final_residual = F.norm();
        
        // Check convergence
        if (final_residual < config_.tolerance) {
            if (config_.verbose) {
                std::cout << "Converged at iteration " << iter << std::endl;
            }
            
            // Update network state
            update_network_state(p, node_ids);
            return true;
        }
        
        // Solve for pressure correction
        Eigen::SparseLU<SparseMatrix> solver;
        solver.compute(J);
        
        if (solver.info() != Eigen::Success) {
            std::cerr << "ERROR: Matrix factorization failed" << std::endl;
            return false;
        }
        
        Vector dp = solver.solve(-F);
        
        if (solver.info() != Eigen::Success) {
            std::cerr << "ERROR: Linear solve failed" << std::endl;
            return false;
        }
        
        // Update pressures with damping
        Real damping = config_.pressure_damping;
        p += damping * dp;
        
        // Display progress
        if (config_.verbose && (iter < 5 || iter % 10 == 0)) {
            Real max_dp = dp.cwiseAbs().maxCoeff();
            std::cout << "Iter " << std::setw(3) << iter 
                      << ": max_change = " << std::scientific << std::setprecision(3) << max_dp
                      << ", rms_change = " << dp.norm() / std::sqrt(n) << std::endl;
        }
    }
    
    // Not converged
    update_network_state(p, node_ids);
    return false;
}

bool SteadyStateSolver::build_jacobian_and_residual(
    SparseMatrix& J, Vector& F, const Vector& p, 
    const std::vector<std::string>& node_ids) {
    
    std::vector<Eigen::Triplet<Real>> triplets;
    F.setZero();
    
    const auto& nodes = network_->nodes();
    const auto& pipes = network_->pipes();
    
    // For each node, write mass balance equation
    for (size_t i = 0; i < node_ids.size(); ++i) {
        const std::string& node_id = node_ids[i];
        
        // Check if pressure is specified
        if (network_->pressure_specs().count(node_id) > 0) {
            // Dirichlet BC: p_i = p_specified
            triplets.push_back(Eigen::Triplet<Real>(static_cast<int>(i), static_cast<int>(i), 1.0));
            F(static_cast<int>(i)) = p(static_cast<int>(i)) - network_->pressure_specs().at(node_id);
        } else {
            // Mass balance: S(flows in) - S(flows out) = Q_specified
            Real Q_specified = network_->flow_specs().count(node_id) > 0 ? 
                              network_->flow_specs().at(node_id) : 0.0;
            
            // Process all connected pipes
            for (const auto& [pipe_id, pipe] : pipes) {
                if (pipe->upstream()->id() == node_id) {
                    // Flow out of node
                    size_t j = std::find(node_ids.begin(), node_ids.end(), 
                                       pipe->downstream()->id()) - node_ids.begin();
                    
                    Real pi = p(static_cast<int>(i));
                    Real pj = p(static_cast<int>(j));
                    
                    // Calculate flow and derivatives
                    Real Q, dQ_dpi, dQ_dpj;
                    calculate_pipe_flow_and_derivatives(pipe, pi, pj, Q, dQ_dpi, dQ_dpj);
                    
                    F(static_cast<int>(i)) -= Q;  // Flow out
                    triplets.push_back(Eigen::Triplet<Real>(static_cast<int>(i), static_cast<int>(i), -dQ_dpi));
                    triplets.push_back(Eigen::Triplet<Real>(static_cast<int>(i), static_cast<int>(j), -dQ_dpj));
                    
                } else if (pipe->downstream()->id() == node_id) {
                    // Flow into node
                    size_t j = std::find(node_ids.begin(), node_ids.end(), 
                                       pipe->upstream()->id()) - node_ids.begin();
                    
                    Real pi = p(static_cast<int>(i));
                    Real pj = p(static_cast<int>(j));
                    
                    // Calculate flow and derivatives
                    Real Q, dQ_dpj, dQ_dpi;
                    calculate_pipe_flow_and_derivatives(pipe, pj, pi, Q, dQ_dpj, dQ_dpi);
                    
                    F(static_cast<int>(i)) += Q;  // Flow in
                    triplets.push_back(Eigen::Triplet<Real>(static_cast<int>(i), static_cast<int>(j), dQ_dpj));
                    triplets.push_back(Eigen::Triplet<Real>(static_cast<int>(i), static_cast<int>(i), dQ_dpi));
                }
            }
            
            F(static_cast<int>(i)) -= Q_specified;
        }
    }
    
    // Build sparse matrix
    J.setFromTriplets(triplets.begin(), triplets.end());
    J.makeCompressed();
    
    return true;
}

void SteadyStateSolver::calculate_pipe_flow_and_derivatives(
    const Ptr<Pipe>& pipe, Real p_upstream, Real p_downstream,
    Real& Q, Real& dQ_dp_upstream, Real& dQ_dp_downstream) {
    
    // Get pipe properties
    Real L = pipe->length();
    Real D = pipe->diameter();
    Real A = pipe->area();
    Real e = pipe->roughness();
    
    // Get fluid properties
    Real rho = fluid_.mixture_density();
    Real mu = fluid_.mixture_viscosity();
    
    // Include elevation effects
    Real z_up = pipe->upstream()->elevation();
    Real z_down = pipe->downstream()->elevation();
    Real dp = p_upstream - p_downstream - rho * constants::GRAVITY * (z_down - z_up);
    
    // Handle near-zero pressure drop
    if (std::abs(dp) < 1e-10) {
        Q = 0.0;
        dQ_dp_upstream = 1e-6;  // Small non-zero value for numerical stability
        dQ_dp_downstream = -1e-6;
        return;
    }
    
    // Estimate velocity for Reynolds number
    Real v_est = std::sqrt(2.0 * std::abs(dp) / rho);
    Real Re = rho * v_est * D / mu;
    
    // Calculate friction factor
    Real f;
    if (Re < 2300) {
        // Laminar flow
        f = 64.0 / std::max(Re, 1.0);
    } else {
        // Turbulent flow - Swamee-Jain approximation
        Real a = -2.0 * std::log10(e/(3.7*D) + 5.74/std::pow(Re, 0.9));
        f = 0.25 / (a * a);
    }
    
    // Calculate flow rate
    // Q = sign(dp) * A * sqrt(2*|dp|/(f*L/D*rho))
    Real K = A * std::sqrt(2.0 * D / (f * L * rho));
    Q = (dp > 0 ? 1 : -1) * K * std::sqrt(std::abs(dp));
    
    // Calculate derivatives
    Real dQ_d_dp = 0.5 * K / std::sqrt(std::abs(dp));
    dQ_dp_upstream = (dp > 0 ? 1 : -1) * dQ_d_dp;
    dQ_dp_downstream = -dQ_dp_upstream;
}

void SteadyStateSolver::update_network_state(const Vector& p, const std::vector<std::string>& node_ids) {
    const auto& nodes = network_->nodes();
    const auto& pipes = network_->pipes();
    
    // Update node pressures
    for (size_t i = 0; i < node_ids.size(); ++i) {
        nodes.at(node_ids[i])->set_pressure(p(static_cast<int>(i)));
    }
    
    // Calculate and update pipe flows
    for (const auto& [pipe_id, pipe] : pipes) {
        Real p_up = pipe->upstream()->pressure();
        Real p_down = pipe->downstream()->pressure();
        
        Real Q, dQ_dp_up, dQ_dp_down;
        calculate_pipe_flow_and_derivatives(pipe, p_up, p_down, Q, dQ_dp_up, dQ_dp_down);
        
        pipe->set_flow_rate(Q);
        pipe->set_velocity(Q / pipe->area());
    }
}

void SteadyStateSolver::extract_results(SolutionResults& results) {
    // Store node pressures and temperatures
    for (const auto& [id, node] : network_->nodes()) {
        results.node_pressures[id] = node->pressure();
        results.node_temperatures[id] = node->temperature();
    }
    
    // Store pipe results
    for (const auto& [id, pipe] : network_->pipes()) {
        results.pipe_flow_rates[id] = pipe->flow_rate();
        results.pipe_pressure_drops[id] = pipe->upstream()->pressure() - pipe->downstream()->pressure();
        
        // Calculate additional properties if needed
        if (results.pipe_velocities.empty()) {  // Only if map exists
            results.pipe_velocities[id] = pipe->velocity();
        }
        
        if (results.pipe_reynolds_numbers.empty()) {  // Only if map exists
            Real Re = pipe->reynolds_number(fluid_.mixture_viscosity(), fluid_.mixture_density());
            results.pipe_reynolds_numbers[id] = Re;
        }
    }
}

// Legacy interface implementations (for backward compatibility)
void SteadyStateSolver::build_system_matrix(SparseMatrix& A, Vector& b) {
    // Not used in new implementation
}

void SteadyStateSolver::apply_boundary_conditions(SparseMatrix& A, Vector& b) {
    // Not used in new implementation
}

void SteadyStateSolver::update_solution(const Vector& x) {
    // Not used in new implementation
}

bool SteadyStateSolver::check_convergence(const Vector& residual) {
    return residual.norm() < config_.tolerance;
}

Real SteadyStateSolver::calculate_pressure_drop(const Ptr<Pipe>& pipe) {
    return pipe->upstream()->pressure() - pipe->downstream()->pressure();
}

} // namespace pipeline_sim
