// ===== solver.cpp - Commercial-Grade Implementation =====
#include "pipeline_sim/solver.h"
#include <Eigen/SparseLU>
#include <Eigen/IterativeLinearSolvers>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <exception>
#include <numeric>

// Define M_PI for Windows compatibility
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace pipeline_sim {

// ===== Base Solver Implementation =====
Solver::Solver(Ptr<Network> network, const FluidProperties& fluid)
    : network_(network), fluid_(fluid) {
}

SolutionResults Solver::solve() {
    SolutionResults results;
    results.converged = false;
    results.iterations = 0;
    results.residual = 1e10;
    results.computation_time = 0.0;
    return results;
}

void Solver::reset() {
    prev_pressures_.clear();
    prev_flows_.clear();
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

// ===== Commercial-Grade SteadyStateSolver Implementation =====
SolutionResults SteadyStateSolver::solve() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    SolutionResults results;
    results.converged = false;
    results.iterations = 0;
    results.residual = 1e10;
    
    // Validate network
    if (!network_ || network_->nodes().empty()) {
        std::cerr << "ERROR: Invalid network" << std::endl;
        return results;
    }
    
    // Check for at least one pressure BC
    if (network_->pressure_specs().empty()) {
        std::cerr << "ERROR: At least one pressure boundary condition required" << std::endl;
        return results;
    }
    
    if (config_.verbose) {
        std::cout << "\n=== Starting Commercial-Grade Steady-State Solver ===" << std::endl;
        std::cout << "Network: " << network_->nodes().size() << " nodes, " 
                  << network_->pipes().size() << " pipes" << std::endl;
        std::cout << "Pressure BCs: " << network_->pressure_specs().size() 
                  << ", Flow BCs: " << network_->flow_specs().size() << std::endl;
        std::cout << "Fluid: density = " << fluid_.mixture_density() 
                  << " kg/m³, viscosity = " << fluid_.mixture_viscosity() * 1000 << " cP" << std::endl;
    }
    
    // Solve using unified formulation
    bool converged = false;
    int iterations_performed = 0;
    Real final_residual = 1e10;
    
    try {
        converged = solve_unified_formulation(iterations_performed, final_residual);
    } catch (const std::exception& e) {
        std::cerr << "ERROR in solver: " << e.what() << std::endl;
    }
    
    // Update results
    results.converged = converged;
    results.iterations = iterations_performed;
    results.residual = final_residual;
    
    // Extract detailed results
    extract_detailed_results(results);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    results.computation_time = std::chrono::duration<Real>(end_time - start_time).count();
    
    if (config_.verbose) {
        std::cout << "\nSolver completed:" << std::endl;
        std::cout << "  Converged: " << (converged ? "YES" : "NO") << std::endl;
        std::cout << "  Iterations: " << iterations_performed << std::endl;
        std::cout << "  Final residual: " << std::scientific << final_residual << std::endl;
        std::cout << "  Max mass imbalance: " << results.max_mass_imbalance << " kg/s" << std::endl;
        std::cout << "  Time: " << std::fixed << std::setprecision(3) 
                  << results.computation_time << " seconds" << std::endl;
    }
    
    // Store solution for next warm start
    if (config_.use_previous_solution && converged) {
        prev_pressures_ = results.node_pressures;
        prev_flows_ = results.pipe_flow_rates;
    }
    
    return results;
}

bool SteadyStateSolver::solve_unified_formulation(int& iterations_performed, Real& final_residual) {
    const auto& nodes = network_->nodes();
    const auto& pipes = network_->pipes();
    
    // Build variable list: pressures for nodes without pressure BC
    std::vector<std::string> var_names;
    std::map<std::string, size_t> var_indices;
    size_t var_count = 0;
    
    // Add pressure variables for nodes without pressure BC
    for (const auto& [node_id, node] : nodes) {
        if (network_->pressure_specs().count(node_id) == 0) {
            var_names.push_back("P_" + node_id);
            var_indices["P_" + node_id] = var_count++;
        }
    }
    
    // If all pressures are specified, we have a trivial problem
    if (var_names.empty()) {
        if (config_.verbose) {
            std::cout << "All pressures specified - calculating flows directly" << std::endl;
        }
        
        // Just calculate flows and we're done
        for (const auto& [pipe_id, pipe] : pipes) {
            Real p_up = pipe->upstream()->pressure();
            Real p_down = pipe->downstream()->pressure();
            Real Q, dQ_dup, dQ_ddown, f;
            calculate_pipe_flow_advanced(pipe, p_up, p_down, Q, dQ_dup, dQ_ddown, f);
            pipe->set_flow_rate(Q);
            pipe->set_velocity(Q / pipe->area());
        }
        
        iterations_performed = 1;
        final_residual = 0.0;
        return true;
    }
    
    size_t n = var_names.size();
    
    // Initialize solution vector
    Vector x(n);
    initialize_solution_advanced(x, var_names, var_indices);
    
    // Newton-Raphson iteration
    Real prev_residual = 1e10;
    
    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        iterations_performed = iter + 1;
        
        // Build Jacobian and residual
        SparseMatrix J(n, n);
        Vector F(n);
        
        if (!build_unified_jacobian(J, F, x, var_names, var_indices)) {
            std::cerr << "ERROR: Failed to build system" << std::endl;
            return false;
        }
        
        // Compute residual norm
        final_residual = F.norm();
        Real max_residual = F.cwiseAbs().maxCoeff();
        
        // Check convergence
        if (final_residual < config_.tolerance && max_residual < config_.tolerance) {
            if (config_.verbose) {
                std::cout << "Converged at iteration " << iter << std::endl;
            }
            
            // Update network state
            update_network_from_solution(x, var_names, var_indices);
            return true;
        }
        
        // Solve for correction
        Eigen::SparseLU<SparseMatrix> solver;
        solver.compute(J);
        
        // Calculate damping factor for this iteration
        Real damping_factor = calculate_adaptive_damping(config_.pressure_damping, iter, 
                                                        final_residual, prev_residual);
        
        if (solver.info() != Eigen::Success) {
            // Try BiCGSTAB if LU fails
            if (config_.verbose) {
                std::cout << "LU decomposition failed, trying iterative solver" << std::endl;
            }
            
            Eigen::BiCGSTAB<SparseMatrix> iterative_solver;
            iterative_solver.compute(J);
            Vector dx = iterative_solver.solve(-F);
            
            if (iterative_solver.info() != Eigen::Success) {
                std::cerr << "ERROR: Linear solve failed" << std::endl;
                return false;
            }
            
            x += damping_factor * dx;
        } else {
            Vector dx = solver.solve(-F);
            
            // Apply adaptive damping
            x += damping_factor * dx;
        }
        
        // Enforce bounds
        enforce_solution_bounds(x, var_names);
        
        // Display progress
        if (config_.verbose && (iter < 5 || iter % 10 == 0)) {
            std::cout << "Iter " << std::setw(3) << iter 
                      << ": residual = " << std::scientific << std::setprecision(3) << final_residual
                      << ", max_residual = " << max_residual
                      << ", damping = " << std::fixed << std::setprecision(2) << damping_factor << std::endl;
        }
        
        prev_residual = final_residual;
    }
    
    // Not converged
    update_network_from_solution(x, var_names, var_indices);
    return false;
}

bool SteadyStateSolver::build_unified_jacobian(
    SparseMatrix& J, Vector& F,
    const Vector& x,
    const std::vector<std::string>& var_names,
    const std::map<std::string, size_t>& var_indices) {
    
    std::vector<Eigen::Triplet<Real>> triplets;
    F.setZero();
    
    const auto& nodes = network_->nodes();
    const auto& pipes = network_->pipes();
    
    // First, update network state with current solution
    update_network_from_solution(x, var_names, var_indices);
    
    // For each node without pressure BC, write mass balance equation
    size_t eq_idx = 0;
    for (const auto& [node_id, node] : nodes) {
        if (network_->pressure_specs().count(node_id) > 0) {
            continue;  // Skip nodes with pressure BC
        }
        
        // This node needs a mass balance equation
        Real specified_flow = network_->flow_specs().count(node_id) > 0 ?
                             network_->flow_specs().at(node_id) : 0.0;
        
        Real total_flow_in = 0.0;
        
        // Process all connected pipes
        for (const auto& [pipe_id, pipe] : pipes) {
            Real Q = 0.0;
            Real dQ_dp_up = 0.0;
            Real dQ_dp_down = 0.0;
            Real f = 0.0;
            
            if (pipe->upstream()->id() == node_id) {
                // Flow out of node (negative contribution)
                Real p_up = node->pressure();
                Real p_down = pipe->downstream()->pressure();
                
                calculate_pipe_flow_advanced(pipe, p_up, p_down, Q, dQ_dp_up, dQ_dp_down, f);
                
                total_flow_in -= Q;
                
                // Add Jacobian entries
                if (var_indices.count("P_" + node_id) > 0) {
                    size_t col_idx = var_indices.at("P_" + node_id);
                    triplets.push_back(Eigen::Triplet<Real>(
                        static_cast<int>(eq_idx), static_cast<int>(col_idx), -dQ_dp_up));
                }
                
                if (var_indices.count("P_" + pipe->downstream()->id()) > 0) {
                    size_t col_idx = var_indices.at("P_" + pipe->downstream()->id());
                    triplets.push_back(Eigen::Triplet<Real>(
                        static_cast<int>(eq_idx), static_cast<int>(col_idx), -dQ_dp_down));
                }
                
            } else if (pipe->downstream()->id() == node_id) {
                // Flow into node (positive contribution)
                Real p_up = pipe->upstream()->pressure();
                Real p_down = node->pressure();
                
                calculate_pipe_flow_advanced(pipe, p_up, p_down, Q, dQ_dp_up, dQ_dp_down, f);
                
                total_flow_in += Q;
                
                // Add Jacobian entries
                if (var_indices.count("P_" + pipe->upstream()->id()) > 0) {
                    size_t col_idx = var_indices.at("P_" + pipe->upstream()->id());
                    triplets.push_back(Eigen::Triplet<Real>(
                        static_cast<int>(eq_idx), static_cast<int>(col_idx), dQ_dp_up));
                }
                
                if (var_indices.count("P_" + node_id) > 0) {
                    size_t col_idx = var_indices.at("P_" + node_id);
                    triplets.push_back(Eigen::Triplet<Real>(
                        static_cast<int>(eq_idx), static_cast<int>(col_idx), dQ_dp_down));
                }
            }
        }
        
        // Mass balance residual: flow_in - specified_flow = 0
        F(static_cast<int>(eq_idx)) = total_flow_in - specified_flow;
        eq_idx++;
    }
    
    // Build sparse matrix
    J.setFromTriplets(triplets.begin(), triplets.end());
    J.makeCompressed();
    
    return true;
}

void SteadyStateSolver::calculate_pipe_flow_advanced(
    const Ptr<Pipe>& pipe, Real p_upstream, Real p_downstream,
    Real& Q, Real& dQ_dp_up, Real& dQ_dp_down, Real& friction_factor) {
    
    // Get pipe and fluid properties
    Real L = pipe->length();
    Real D = pipe->diameter();
    Real A = pipe->area();
    Real e = pipe->roughness();
    Real rho = fluid_.mixture_density();
    Real mu = fluid_.mixture_viscosity();
    
    // Include elevation effects
    Real z_up = pipe->upstream()->elevation();
    Real z_down = pipe->downstream()->elevation();
    Real dp = p_upstream - p_downstream - rho * constants::GRAVITY * (z_down - z_up);
    
    // Handle near-zero pressure drop
    if (std::abs(dp) < config_.jacobian_epsilon) {
        Q = 0.0;
        // Use finite difference approximation for derivatives
        Real dp_perturb = config_.jacobian_epsilon;
        Real Q_plus, dQ_dummy1, dQ_dummy2, f_dummy;
        calculate_pipe_flow_advanced(pipe, p_upstream + dp_perturb, p_downstream,
                                   Q_plus, dQ_dummy1, dQ_dummy2, f_dummy);
        dQ_dp_up = Q_plus / dp_perturb;
        dQ_dp_down = -dQ_dp_up;
        friction_factor = 0.02;  // Default
        return;
    }
    
    // Initial estimate of velocity
    Real v_est = std::sqrt(2.0 * std::abs(dp) / rho);
    v_est = std::max(v_est, config_.min_flow_velocity);
    
    // Iterative solution for flow considering variable friction factor
    Real f = 0.02;  // Initial guess
    Real Q_prev = 0.0;
    
    for (int i = 0; i < 10; ++i) {
        // Current velocity estimate
        Real v = v_est;
        Real Re = rho * v * D / mu;
        Re = std::max(Re, 1.0);  // Avoid zero Reynolds
        
        // Calculate friction factor with derivative
        Real df_dRe = 0.0;
        f = calculate_friction_factor_advanced(Re, e/D, df_dRe);
        
        // Apply laminar correction if needed
        if (config_.enable_laminar_correction) {
            Real dCorr_dRe = 0.0;
            Real correction = laminar_correction_factor(Re, dCorr_dRe);
            f *= correction;
            df_dRe = df_dRe * correction + f * dCorr_dRe / correction;
        }
        
        // Calculate flow
        Real K = A * std::sqrt(2.0 * D / (f * L * rho));
        Q = (dp > 0 ? 1 : -1) * K * std::sqrt(std::abs(dp));
        
        // Check convergence
        if (std::abs(Q - Q_prev) < 1e-10) {
            break;
        }
        
        Q_prev = Q;
        v_est = std::abs(Q) / A;
    }
    
    // Calculate derivatives
    if (std::abs(Q) > config_.min_flow_velocity * A) {
        Real v = std::abs(Q) / A;
        Real Re = rho * v * D / mu;
        
        // Basic derivative
        Real K = A * std::sqrt(2.0 * D / (f * L * rho));
        Real dQ_d_dp = 0.5 * K / std::sqrt(std::abs(dp));
        
        // Correction for friction factor variation with Reynolds number
        Real df_dRe = 0.0;
        calculate_friction_factor_advanced(Re, e/D, df_dRe);
        
        Real dRe_dQ = rho * D / (mu * A);
        Real df_dQ = df_dRe * dRe_dQ;
        Real dQ_d_dp_corrected = dQ_d_dp / (1 + 0.5 * Q * df_dQ / f);
        
        dQ_dp_up = (dp > 0 ? 1 : -1) * dQ_d_dp_corrected;
        dQ_dp_down = -dQ_dp_up;
    } else {
        // Near zero flow - use numerical derivatives
        Real eps = config_.jacobian_epsilon;
        Real Q_plus, Q_minus, dummy1, dummy2, f_dummy;
        
        calculate_pipe_flow_advanced(pipe, p_upstream + eps, p_downstream,
                                   Q_plus, dummy1, dummy2, f_dummy);
        calculate_pipe_flow_advanced(pipe, p_upstream - eps, p_downstream,
                                   Q_minus, dummy1, dummy2, f_dummy);
        dQ_dp_up = (Q_plus - Q_minus) / (2 * eps);
        
        calculate_pipe_flow_advanced(pipe, p_upstream, p_downstream + eps,
                                   Q_plus, dummy1, dummy2, f_dummy);
        calculate_pipe_flow_advanced(pipe, p_upstream, p_downstream - eps,
                                   Q_minus, dummy1, dummy2, f_dummy);
        dQ_dp_down = (Q_plus - Q_minus) / (2 * eps);
    }
    
    friction_factor = f;
}

Real SteadyStateSolver::calculate_friction_factor_advanced(
    Real Re, Real relative_roughness, Real& df_dRe) {
    
    if (Re < 1.0) {
        // Avoid numerical issues
        Real f = 64.0;
        df_dRe = -64.0;
        return f;
    }
    
    if (Re < config_.laminar_transition_Re) {
        // Laminar flow
        Real f = 64.0 / Re;
        df_dRe = -64.0 / (Re * Re);
        return f;
    }
    
    // Turbulent flow - use Colebrook-White
    Real f = solve_colebrook_white(Re, relative_roughness);
    
    // Approximate derivative using Swamee-Jain
    Real a = -2.0 * std::log10(relative_roughness/3.7 + 5.74/std::pow(Re, 0.9));
    Real da_dRe = 2.0 * 5.74 * 0.9 / (std::pow(Re, 1.9) * std::log(10.0) * 
                  (relative_roughness/3.7 + 5.74/std::pow(Re, 0.9)));
    
    df_dRe = -0.5 * da_dRe / (a * a * a);
    
    return f;
}

Real SteadyStateSolver::solve_colebrook_white(
    Real Re, Real relative_roughness, Real initial_guess) {
    
    // Use Swamee-Jain as a very good approximation
    Real a = -2.0 * std::log10(relative_roughness/3.7 + 5.74/std::pow(Re, 0.9));
    return 0.25 / (a * a);
    
    // For even higher accuracy, could iterate:
    /*
    Real f = initial_guess;
    for (int i = 0; i < 5; ++i) {
        Real f_new = 1.0 / std::pow(-2.0 * std::log10(relative_roughness/3.7 + 
                                    2.51/(Re * std::sqrt(f))), 2.0);
        if (std::abs(f_new - f) < 1e-10) break;
        f = f_new;
    }
    return f;
    */
}

Real SteadyStateSolver::laminar_correction_factor(Real Re, Real& dFactor_dRe) {
    // Smooth transition between laminar and turbulent
    if (!config_.enable_laminar_correction) {
        dFactor_dRe = 0.0;
        return 1.0;
    }
    
    Real Re_trans = config_.laminar_transition_Re;
    Real zone = Re_trans * (config_.critical_zone_factor - 1.0);
    
    if (Re < Re_trans - zone) {
        // Fully laminar
        dFactor_dRe = 0.0;
        return 1.0;
    } else if (Re > Re_trans + zone) {
        // Fully turbulent
        dFactor_dRe = 0.0;
        return 1.0;
    } else {
        // Transition zone - smooth interpolation
        Real x = (Re - (Re_trans - zone)) / (2.0 * zone);
        Real factor = 0.5 * (1.0 + std::sin(M_PI * (x - 0.5)));
        dFactor_dRe = M_PI * std::cos(M_PI * (x - 0.5)) / (4.0 * zone);
        return factor;
    }
}

void SteadyStateSolver::update_network_from_solution(
    const Vector& x,
    const std::vector<std::string>& var_names,
    const std::map<std::string, size_t>& var_indices) {
    
    const auto& nodes = network_->nodes();
    const auto& pipes = network_->pipes();
    
    // Update node pressures from solution
    for (size_t i = 0; i < var_names.size(); ++i) {
        if (var_names[i].substr(0, 2) == "P_") {
            std::string node_id = var_names[i].substr(2);
            if (nodes.count(node_id) > 0) {
                nodes.at(node_id)->set_pressure(x(static_cast<int>(i)));
            }
        }
    }
    
    // Calculate and update all pipe flows
    for (const auto& [pipe_id, pipe] : pipes) {
        Real p_up = pipe->upstream()->pressure();
        Real p_down = pipe->downstream()->pressure();
        
        Real Q, dQ_up, dQ_down, f;
        calculate_pipe_flow_advanced(pipe, p_up, p_down, Q, dQ_up, dQ_down, f);
        
        pipe->set_flow_rate(Q);
        pipe->set_velocity(Q / pipe->area());
    }
}

void SteadyStateSolver::extract_detailed_results(SolutionResults& results) {
    const auto& nodes = network_->nodes();
    const auto& pipes = network_->pipes();
    
    // Node results
    for (const auto& [id, node] : nodes) {
        results.node_pressures[id] = node->pressure();
        results.node_temperatures[id] = node->temperature();
        
        // Calculate mass imbalance
        Real imbalance = 0.0;
        for (const auto& [pipe_id, pipe] : pipes) {
            if (pipe->upstream()->id() == id) {
                imbalance -= pipe->flow_rate() * fluid_.mixture_density();
            } else if (pipe->downstream()->id() == id) {
                imbalance += pipe->flow_rate() * fluid_.mixture_density();
            }
        }
        
        // Account for specified flows
        if (network_->flow_specs().count(id) > 0) {
            imbalance -= network_->flow_specs().at(id) * fluid_.mixture_density();
        }
        
        results.node_mass_imbalance[id] = imbalance;
        results.max_mass_imbalance = std::max(results.max_mass_imbalance, std::abs(imbalance));
    }
    
    // Pipe results
    for (const auto& [id, pipe] : pipes) {
        results.pipe_flow_rates[id] = pipe->flow_rate();
        results.pipe_velocities[id] = pipe->velocity();
        results.pipe_pressure_drops[id] = pipe->upstream()->pressure() - 
                                         pipe->downstream()->pressure();
        
        Real Re = pipe->reynolds_number(fluid_.mixture_viscosity(), fluid_.mixture_density());
        results.pipe_reynolds_numbers[id] = Re;
        
        Real df_dRe = 0.0;
        results.pipe_friction_factors[id] = calculate_friction_factor_advanced(
            Re, pipe->roughness() / pipe->diameter(), df_dRe);
    }
}

void SteadyStateSolver::enforce_solution_bounds(
    Vector& x, const std::vector<std::string>& var_names) {
    
    for (size_t i = 0; i < var_names.size(); ++i) {
        if (var_names[i].substr(0, 2) == "P_") {
            // Pressure variable
            x(static_cast<int>(i)) = std::max(config_.min_pressure, 
                                             std::min(config_.max_pressure, x(static_cast<int>(i))));
        }
    }
}

Real SteadyStateSolver::calculate_adaptive_damping(
    Real base_damping, int iteration, Real current_residual, Real prev_residual) {
    
    if (!config_.adaptive_damping) {
        return base_damping;
    }
    
    // Increase damping if residual is increasing
    if (current_residual > prev_residual && iteration > 0) {
        return base_damping * 0.5;
    }
    
    // Decrease damping if converging well
    if (current_residual < 0.1 * prev_residual && iteration > 5) {
        return std::min(1.0, base_damping * 1.2);
    }
    
    return base_damping;
}

void SteadyStateSolver::initialize_solution_advanced(
    Vector& x,
    const std::vector<std::string>& var_names,
    const std::map<std::string, size_t>& var_indices) {
    
    const auto& nodes = network_->nodes();
    
    // Use previous solution if available
    if (config_.use_previous_solution && !prev_pressures_.empty()) {
        for (size_t i = 0; i < var_names.size(); ++i) {
            if (var_names[i].substr(0, 2) == "P_") {
                std::string node_id = var_names[i].substr(2);
                if (prev_pressures_.count(node_id) > 0) {
                    x(static_cast<int>(i)) = prev_pressures_.at(node_id);
                    continue;
                }
            }
        }
    }
    
    // Smart initialization based on network topology
    // Start with average of known pressures
    Real avg_pressure = 0.0;
    int count = 0;
    
    for (const auto& [node_id, pressure] : network_->pressure_specs()) {
        avg_pressure += pressure;
        count++;
    }
    
    if (count > 0) {
        avg_pressure /= count;
    } else {
        avg_pressure = constants::STANDARD_PRESSURE;
    }
    
    // Initialize unknown pressures
    for (size_t i = 0; i < var_names.size(); ++i) {
        if (var_names[i].substr(0, 2) == "P_") {
            std::string node_id = var_names[i].substr(2);
            
            // Use slightly below average pressure for sinks, above for sources
            if (nodes.at(node_id)->type() == NodeType::SINK) {
                x(static_cast<int>(i)) = avg_pressure * 0.8;
            } else if (nodes.at(node_id)->type() == NodeType::SOURCE) {
                x(static_cast<int>(i)) = avg_pressure * 1.2;
            } else {
                x(static_cast<int>(i)) = avg_pressure;
            }
        }
    }
}

void SteadyStateSolver::reset() {
    Solver::reset();
}

// Legacy implementations (maintained for compatibility)
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
