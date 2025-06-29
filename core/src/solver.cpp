#include "pipeline_sim/solver.h"
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>
#include <Eigen/IterativeLinearSolvers>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace pipeline_sim {

// SolutionResults methods
double SolutionResults::pressure_drop(const std::shared_ptr<Pipe>& pipe) const {
    auto it = pipe_pressure_drops.find(pipe->id());
    return (it != pipe_pressure_drops.end()) ? it->second : 0.0;
}

double SolutionResults::outlet_pressure(const std::shared_ptr<Pipe>& pipe) const {
    auto it = node_pressures.find(pipe->downstream()->id());
    return (it != node_pressures.end()) ? it->second : 0.0;
}

// Base Solver
Solver::Solver(std::shared_ptr<Network> network, const FluidProperties& fluid)
    : network_(network), fluid_(fluid) {
}

// SteadyStateSolver implementation
SteadyStateSolver::SteadyStateSolver(std::shared_ptr<Network> network, const FluidProperties& fluid)
    : Solver(network, fluid) {
    setupSolver();
}

void SteadyStateSolver::setupSolver() {
    // Any additional setup
}

SolutionResults SteadyStateSolver::solve() {
    auto start_time = std::chrono::high_resolution_clock::now();
    SolutionResults results;
    
    if (!network_ || network_->nodes().empty() || network_->pipes().empty()) {
        results.converged = false;
        results.convergence_reason = "Invalid network";
        return results;
    }
    
    // Professional solver setup
    if (config_.verbose) {
        printSolverHeader();
    }
    
    // Initialize solution with smart initial guess
    initializeSolutionProfessional(results);
    
    // Build node and pipe indexing for matrix assembly
    buildSystemIndexing();
    
    // Validate network connectivity and boundary conditions
    if (!validateNetwork()) {
        results.converged = false;
        results.convergence_reason = "Invalid network configuration";
        return results;
    }
    
    // Main Newton-Raphson iteration loop
    bool converged = false;
    double prev_residual = 1e20;
    std::vector<double> recent_residuals;
    
    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        // Step 1: Assemble system of equations
        Eigen::SparseMatrix<double> J;  // Jacobian matrix
        Eigen::VectorXd F;              // Residual vector
        
        assembleSystemOfEquations(J, F, results);
        
        // Calculate current residual
        double current_residual = F.norm() / std::sqrt(F.size());
        results.residual_history.push_back(current_residual);
        
        // Check convergence
        if (checkConvergence(current_residual, prev_residual, recent_residuals, iter)) {
            converged = true;
            results.converged = true;
            results.iterations = iter + 1;
            results.residual = current_residual;
            results.convergence_reason = "Tolerance achieved";
            
            if (config_.verbose) {
                printf("\n? Converged in %d iterations! Final residual: %.2e\n", iter + 1, current_residual);
            }
            break;
        }
        
        // Step 2: Solve linear system for pressure corrections
        Eigen::VectorXd delta_p;
        bool solve_success = solveLinearSystem(J, F, delta_p, results);
        
        if (!solve_success) {
            if (config_.verbose) {
                printf("\n? Linear solver failed at iteration %d\n", iter + 1);
            }
            results.convergence_reason = "Linear solver failure";
            break;
        }
        
        // Step 3: Apply pressure corrections with line search or trust region
        double step_size = 1.0;
        if (config_.use_line_search) {
            step_size = performLineSearch(delta_p, F, current_residual, results);
        } else if (config_.use_trust_region) {
            step_size = applyTrustRegion(delta_p, current_residual);
        } else if (config_.use_adaptive_relaxation) {
            step_size = calculateAdaptiveRelaxation(current_residual, prev_residual, iter);
        } else {
            step_size = config_.relaxation_factor;
        }
        
        results.step_size_history.push_back(step_size);
        
        // Update pressures
        updatePressures(delta_p, step_size, results);
        
        // Update pipe flows with new pressures
        updatePipeFlows(results);
        
        // Verbose output
        if (config_.verbose && (iter % 10 == 0 || iter < 5)) {
            printf("Iter %3d: |F|=%.2e, ||?p||=%.2e bar, step=%.3f\n", 
                   iter + 1, current_residual, delta_p.norm()/1e5, step_size);
        }
        
        // Check for divergence
        if (current_residual > 1e10 || std::isnan(current_residual)) {
            if (config_.verbose) {
                printf("\n? Solver diverged at iteration %d\n", iter + 1);
            }
            results.convergence_reason = "Divergence detected";
            break;
        }
        
        prev_residual = current_residual;
        recent_residuals.push_back(current_residual);
        if (recent_residuals.size() > config_.stagnation_check_window) {
            recent_residuals.erase(recent_residuals.begin());
        }
        
        results.iterations = iter + 1;
        results.residual = current_residual;
    }
    
    // Final results calculation
    calculateFinalResults(results);
    
    // Calculate computation time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    results.computation_time = duration.count() / 1000.0;
    
    if (config_.verbose) {
        printSolverSummary(results);
    }
    
    if (!converged && results.convergence_reason.empty()) {
        results.convergence_reason = "Maximum iterations reached";
    }
    
    return results;
}

void SteadyStateSolver::printSolverHeader() {
    printf("\n================================================================================\n");
    printf("  PROFESSIONAL PIPELINE NETWORK SOLVER v2.0\n");
    printf("  Method: Newton-Raphson with %s Jacobian\n", 
           config_.jacobian_method == SolverConfig::ANALYTICAL ? "Analytical" : "Finite Difference");
    printf("  Linear Solver: %s\n", 
           config_.linear_solver == SolverConfig::LU_DECOMPOSITION ? "LU Decomposition" :
           config_.linear_solver == SolverConfig::QR_DECOMPOSITION ? "QR Decomposition" : "BiCGSTAB");
    printf("================================================================================\n");
}

void SteadyStateSolver::printSolverSummary(const SolutionResults& results) {
    printf("\n--------------------------------------------------------------------------------\n");
    printf("  SOLVER SUMMARY\n");
    printf("  Status: %s\n", results.converged ? "CONVERGED" : "NOT CONVERGED");
    printf("  Iterations: %d\n", results.iterations);
    printf("  Final Residual: %.2e\n", results.residual);
    printf("  Computation Time: %.3f seconds\n", results.computation_time);
    printf("  Convergence Reason: %s\n", results.convergence_reason.c_str());
    if (results.jacobian_condition_number > 0) {
        printf("  Jacobian Condition Number: %.2e\n", results.jacobian_condition_number);
    }
    printf("--------------------------------------------------------------------------------\n");
}

void SteadyStateSolver::buildSystemIndexing() {
    unknown_pressure_nodes_.clear();
    node_to_index_.clear();
    node_to_pipes_.clear();
    
    size_t index = 0;
    for (const auto& [node_id, node] : network_->nodes()) {
        if (!node->has_pressure_bc()) {
            unknown_pressure_nodes_.push_back(node_id);
            node_to_index_[node_id] = index++;
        }
        node_to_pipes_[node_id] = std::vector<std::string>();
    }
    
    // Build node-pipe connectivity
    for (const auto& [pipe_id, pipe] : network_->pipes()) {
        node_to_pipes_[pipe->upstream()->id()].push_back(pipe_id);
        node_to_pipes_[pipe->downstream()->id()].push_back(pipe_id);
    }
}

bool SteadyStateSolver::validateNetwork() {
    // Check if we have at least one pressure boundary condition
    bool has_pressure_bc = false;
    for (const auto& [node_id, node] : network_->nodes()) {
        if (node->has_pressure_bc()) {
            has_pressure_bc = true;
            break;
        }
    }
    
    if (!has_pressure_bc) {
        if (config_.verbose) {
            printf("? Error: Network has no pressure boundary conditions\n");
        }
        return false;
    }
    
    // Check network connectivity (simplified - should use graph algorithms)
    return true;
}

void SteadyStateSolver::initializeSolutionProfessional(SolutionResults& results) {
    // Initialize all node pressures
    for (const auto& [id, node] : network_->nodes()) {
        if (node->has_pressure_bc()) {
            results.node_pressures[id] = node->pressure_bc();
            node->set_pressure(node->pressure_bc());
        } else {
            // Smart initial guess: average of all BC pressures
            double avg_pressure = 0.0;
            int count = 0;
            for (const auto& [nid, n] : network_->nodes()) {
                if (n->has_pressure_bc()) {
                    avg_pressure += n->pressure_bc();
                    count++;
                }
            }
            if (count > 0) {
                results.node_pressures[id] = avg_pressure / count;
                node->set_pressure(avg_pressure / count);
            } else {
                results.node_pressures[id] = 50e5;  // 50 bar default
                node->set_pressure(50e5);
            }
        }
        results.node_temperatures[id] = node->temperature();
    }
    
    // Initialize pipe flows based on initial pressures
    updatePipeFlows(results);
}

void SteadyStateSolver::updatePipeFlows(SolutionResults& results) {
    for (const auto& [pipe_id, pipe] : network_->pipes()) {
        calculatePipeFlowProfessional(pipe, results);
    }
}

void SteadyStateSolver::calculatePipeFlowProfessional(const std::shared_ptr<Pipe>& pipe, 
                                                     SolutionResults& results) {
    // Get pressures
    double p_up = results.node_pressures[pipe->upstream()->id()];
    double p_down = results.node_pressures[pipe->downstream()->id()];
    
    // Pipe properties
    double L = pipe->length();
    double D = pipe->diameter();
    double A = M_PI * D * D / 4.0;
    double eps = pipe->roughness();
    
    // Fluid properties
    double rho = fluid_.mixture_density();
    double mu = fluid_.mixture_viscosity();
    
    // Elevation difference
    double z_up = pipe->upstream()->elevation();
    double z_down = pipe->downstream()->elevation();
    double dz = z_down - z_up;
    
    // Total pressure difference (driving force)
    double dp_total = p_up - p_down - rho * 9.81 * dz;
    
    // Initial guess for flow rate
    double Q = pipe->flow_rate();
    if (std::abs(Q) < 1e-10) {
        // Laminar flow estimate
        Q = M_PI * D * D * D * D * dp_total / (128.0 * mu * L);
    }
    
    // Iterative solution for flow rate (implicit friction factor)
    const int max_iter = 50;
    const double tol = 1e-10;
    
    for (int i = 0; i < max_iter; ++i) {
        double v = Q / A;
        double Re = std::abs(rho * v * D / mu);
        
        // Calculate friction factor
        double f = calculateFrictionFactorColebrook(Re, D, eps);
        
        // Calculate pressure drop due to friction
        double dp_friction = 0.0;
        if (std::abs(v) > 1e-10) {
            dp_friction = f * L * rho * v * std::abs(v) / (2.0 * D);
        }
        
        // Function to solve: dp_total - dp_friction = 0
        double F_Q = dp_total - dp_friction;
        
        // Derivative: dF/dQ
        double dF_dQ = 0.0;
        if (std::abs(v) > 1e-10) {
            // Approximate derivative (ignoring df/dRe for stability)
            dF_dQ = -f * L * rho * 2.0 * std::abs(v) / (2.0 * D * A);
            if (v < 0) dF_dQ = -dF_dQ;
        } else {
            // Laminar approximation
            dF_dQ = -128.0 * mu * L / (M_PI * D * D * D * D);
        }
        
        // Newton-Raphson update
        double dQ = -F_Q / dF_dQ;
        
        // Limit step size
        if (std::abs(dQ) > 0.5 * std::abs(Q) + 0.01) {
            dQ = (dQ > 0 ? 1.0 : -1.0) * (0.5 * std::abs(Q) + 0.01);
        }
        
        Q = Q + dQ;
        
        // Check convergence
        if (std::abs(dQ) < tol * (std::abs(Q) + tol)) {
            break;
        }
    }
    
    // Store results
    results.pipe_flow_rates[pipe->id()] = Q;
    pipe->set_flow_rate(Q);
    
    // Calculate derived quantities
    double v = Q / A;
    double Re = std::abs(rho * v * D / mu);
    double f = calculateFrictionFactorColebrook(Re, D, eps);
    
    results.pipe_velocities[pipe->id()] = v;
    results.pipe_reynolds_numbers[pipe->id()] = Re;
    results.pipe_friction_factors[pipe->id()] = f;
    results.pipe_pressure_drops[pipe->id()] = p_up - p_down;
}

double SteadyStateSolver::calculateFrictionFactorColebrook(double Re, double D, double eps) {
    if (Re < 2300.0) {
        // Laminar flow
        return 64.0 / Re;
    }
    
    // Turbulent flow - Colebrook-White equation
    // 1/sqrt(f) = -2*log10(eps/(3.7*D) + 2.51/(Re*sqrt(f)))
    
    double f = 0.02;  // Initial guess
    const int max_iter = 50;
    const double tol = 1e-12;
    
    for (int i = 0; i < max_iter; ++i) {
        double sqrt_f = std::sqrt(f);
        double A = eps / (3.7 * D);
        double B = 2.51 / (Re * sqrt_f);
        
        double g = 1.0 / sqrt_f + 2.0 * std::log10(A + B);
        double dg_df = -0.5 / (f * sqrt_f) - 2.0 / (std::log(10.0) * (A + B)) * 
                       (-2.51 / (2.0 * Re * f * sqrt_f));
        
        double df = -g / dg_df;
        
        // Limit step
        if (df > 0.5 * f) df = 0.5 * f;
        if (df < -0.5 * f) df = -0.5 * f;
        
        f = f + df;
        
        if (std::abs(df) < tol) break;
    }
    
    // Bounds check
    if (f < 0.008) f = 0.008;
    if (f > 0.1) f = 0.1;
    
    return f;
}

void SteadyStateSolver::assembleSystemOfEquations(Eigen::SparseMatrix<double>& J, 
                                                 Eigen::VectorXd& F,
                                                 const SolutionResults& results) {
    size_t n = unknown_pressure_nodes_.size();
    
    // Initialize
    J.resize(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
    F.resize(static_cast<Eigen::Index>(n));
    F.setZero();
    
    std::vector<Eigen::Triplet<double>> triplets;
    
    // For each unknown pressure node, write mass balance equation
    for (size_t i = 0; i < n; ++i) {
        const std::string& node_id = unknown_pressure_nodes_[i];
        auto node = network_->nodes().at(node_id);
        
        // Calculate flow balance (sum of flows = 0)
        double flow_balance = 0.0;
        
        // Add flows from connected pipes
        for (const auto& pipe_id : node_to_pipes_[node_id]) {
            auto pipe = network_->pipes().at(pipe_id);
            double flow = results.pipe_flow_rates.at(pipe_id);
            
            if (pipe->upstream()->id() == node_id) {
                flow_balance -= flow;  // Outflow
            } else {
                flow_balance += flow;  // Inflow
            }
        }
        
        // Add source/sink terms
        if (node->fixed_flow_rate() != 0.0) {
            flow_balance += node->fixed_flow_rate();
        }
        
        // Store residual
        F(static_cast<Eigen::Index>(i)) = -flow_balance;
        
        // Calculate Jacobian entries
        if (config_.jacobian_method == SolverConfig::ANALYTICAL) {
            calculateAnalyticalJacobian(static_cast<int>(i), node_id, triplets, results);
        } else {
            calculateFiniteDifferenceJacobian(static_cast<int>(i), node_id, J, results, F);
        }
    }
    
    if (config_.jacobian_method == SolverConfig::ANALYTICAL) {
        J.setFromTriplets(triplets.begin(), triplets.end());
    }
    
    // Make matrix more stable by adding small diagonal term
    for (size_t i = 0; i < n; ++i) {
        J.coeffRef(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(i)) += 1e-10;
    }
}

void SteadyStateSolver::calculateAnalyticalJacobian(int row, const std::string& node_id,
                                                   std::vector<Eigen::Triplet<double>>& triplets,
                                                   const SolutionResults& results) {
    // For each connected pipe, calculate d(flow)/d(pressure)
    for (const auto& pipe_id : node_to_pipes_[node_id]) {
        auto pipe = network_->pipes().at(pipe_id);
        
        // Get current state
        double Q = results.pipe_flow_rates.at(pipe_id);
        double p_up = results.node_pressures.at(pipe->upstream()->id());
        double p_down = results.node_pressures.at(pipe->downstream()->id());
        
        // Pipe properties
        double L = pipe->length();
        double D = pipe->diameter();
        double A = M_PI * D * D / 4.0;
        double eps = pipe->roughness();
        
        // Fluid properties
        double rho = fluid_.mixture_density();
        double mu = fluid_.mixture_viscosity();
        
        // Calculate derivatives
        double v = Q / A;
        double Re = std::abs(rho * v * D / mu);
        double f = calculateFrictionFactorColebrook(Re, D, eps);
        
        // Linearized conductance: dQ/dp
        double conductance = 0.0;
        
        if (std::abs(Q) > 1e-10) {
            // Turbulent/transitional flow
            double dp_friction = f * L * rho * v * std::abs(v) / (2.0 * D);
            double dp_total = p_up - p_down;
            
            // Approximate conductance (ignoring df/dQ for stability)
            if (std::abs(dp_total) > 1e-3) {
                conductance = std::abs(Q) / std::abs(dp_total);
            } else {
                conductance = A * std::sqrt(2.0 * D / (f * L * rho)) / 2.0;
            }
        } else {
            // Near-zero flow: use laminar approximation
            conductance = M_PI * D * D * D * D / (128.0 * mu * L);
        }
        
        // Add to Jacobian
        bool is_upstream = (pipe->upstream()->id() == node_id);
        
        // Effect on flow balance from pressure at this node
        if (is_upstream) {
            // Increasing upstream pressure increases outflow
            triplets.push_back(Eigen::Triplet<double>(row, row, -conductance));
            
            // Effect from downstream node (if it's unknown)
            auto down_it = node_to_index_.find(pipe->downstream()->id());
            if (down_it != node_to_index_.end()) {
                triplets.push_back(Eigen::Triplet<double>(row, static_cast<int>(down_it->second), conductance));
            }
        } else {
            // Increasing downstream pressure decreases inflow
            triplets.push_back(Eigen::Triplet<double>(row, row, -conductance));
            
            // Effect from upstream node (if it's unknown)
            auto up_it = node_to_index_.find(pipe->upstream()->id());
            if (up_it != node_to_index_.end()) {
                triplets.push_back(Eigen::Triplet<double>(row, static_cast<int>(up_it->second), conductance));
            }
        }
    }
}

void SteadyStateSolver::calculateFiniteDifferenceJacobian(int row, const std::string& node_id,
                                                         Eigen::SparseMatrix<double>& J,
                                                         const SolutionResults& results,
                                                         const Eigen::VectorXd& F_current) {
    // Finite difference approximation of Jacobian
    double h = config_.finite_diff_step;
    
    // For each unknown pressure node
    for (size_t col = 0; col < unknown_pressure_nodes_.size(); ++col) {
        const std::string& perturb_node = unknown_pressure_nodes_[col];
        
        // Perturb pressure
        SolutionResults perturbed_results = results;
        perturbed_results.node_pressures[perturb_node] += h;
        
        // Recalculate flows for affected pipes
        for (const auto& pipe_id : node_to_pipes_[perturb_node]) {
            auto pipe = network_->pipes().at(pipe_id);
            calculatePipeFlowProfessional(pipe, perturbed_results);
        }
        
        // Calculate perturbed flow balance
        double perturbed_balance = 0.0;
        for (const auto& pipe_id : node_to_pipes_[node_id]) {
            auto pipe = network_->pipes().at(pipe_id);
            double flow = perturbed_results.pipe_flow_rates.at(pipe_id);
            
            if (pipe->upstream()->id() == node_id) {
                perturbed_balance -= flow;
            } else {
                perturbed_balance += flow;
            }
        }
        
        // Jacobian entry
        double original_balance = -F_current(row);  // F stores negative of balance
        J.coeffRef(row, static_cast<int>(col)) = -(perturbed_balance - original_balance) / h;
    }
}

bool SteadyStateSolver::solveLinearSystem(const Eigen::SparseMatrix<double>& J,
                                         const Eigen::VectorXd& F,
                                         Eigen::VectorXd& delta_p,
                                         SolutionResults& results) {
    try {
        if (config_.linear_solver == SolverConfig::LU_DECOMPOSITION) {
            Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
            solver.analyzePattern(J);
            solver.factorize(J);
            
            if (solver.info() != Eigen::Success) {
                return false;
            }
            
            delta_p = solver.solve(F);
            
        } else if (config_.linear_solver == SolverConfig::QR_DECOMPOSITION) {
            Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
            solver.analyzePattern(J);
            solver.factorize(J);
            
            if (solver.info() != Eigen::Success) {
                return false;
            }
            
            delta_p = solver.solve(F);
            results.jacobian_rank = static_cast<int>(solver.rank());
            
        } else {
            // BiCGSTAB iterative solver
            Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
            solver.setTolerance(1e-10);
            solver.setMaxIterations(1000);
            solver.compute(J);
            
            if (solver.info() != Eigen::Success) {
                return false;
            }
            
            delta_p = solver.solve(F);
        }
        
        // Check for NaN or infinite values
        if (!delta_p.allFinite()) {
            return false;
        }
        
        // Limit maximum pressure change
        double max_dp = delta_p.cwiseAbs().maxCoeff();
        if (max_dp > 50e5) {  // 50 bar max change
            delta_p *= 50e5 / max_dp;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        if (config_.verbose) {
            printf("Linear solver exception: %s\n", e.what());
        }
        return false;
    }
}

double SteadyStateSolver::performLineSearch(const Eigen::VectorXd& delta_p,
                                           const Eigen::VectorXd& F0,
                                           double residual0,
                                           SolutionResults& results) {
    double alpha = 1.0;
    const double c1 = config_.line_search_alpha;  // Armijo constant
    const double rho = config_.line_search_beta;  // Backtracking factor
    
    // Save original state
    std::map<std::string, double> original_pressures = results.node_pressures;
    
    // Expected decrease (directional derivative)
    double expected_decrease = -F0.dot(delta_p);
    
    if (expected_decrease >= 0) {
        // Not a descent direction
        return config_.relaxation_factor;
    }
    
    // Line search loop
    for (int ls_iter = 0; ls_iter < config_.max_line_search_iter; ++ls_iter) {
        // Try step
        updatePressures(delta_p, alpha, results);
        updatePipeFlows(results);
        
        // Evaluate new residual
        Eigen::VectorXd F_new;
        F_new.resize(F0.size());
        evaluateResidual(F_new, results);
        
        double residual_new = F_new.norm() / std::sqrt(F_new.size());
        
        // Armijo condition
        if (residual_new <= residual0 + c1 * alpha * expected_decrease) {
            // Accept step
            return alpha;
        }
        
        // Restore state and try smaller step
        results.node_pressures = original_pressures;
        alpha *= rho;
        
        if (alpha < 1e-4) {
            // Very small step - accept it anyway
            break;
        }
    }
    
    // Restore and use small step
    results.node_pressures = original_pressures;
    return alpha;
}

double SteadyStateSolver::applyTrustRegion(Eigen::VectorXd& delta_p, double current_residual) {
    double step_norm = delta_p.norm();
    
    if (step_norm > config_.trust_region_radius) {
        // Scale step to fit in trust region
        delta_p *= config_.trust_region_radius / step_norm;
        return config_.trust_region_radius / step_norm;
    }
    
    return 1.0;
}

double SteadyStateSolver::calculateAdaptiveRelaxation(double current_residual, 
                                                     double prev_residual,
                                                     int iteration) {
    double ratio = current_residual / prev_residual;
    
    if (ratio < 0.5) {
        // Excellent progress - increase relaxation
        return std::min(config_.relaxation_factor * 1.2, config_.max_relaxation);
    } else if (ratio < 0.9) {
        // Good progress - keep current
        return config_.relaxation_factor;
    } else if (ratio < 1.1) {
        // Slow progress - decrease slightly
        return std::max(config_.relaxation_factor * 0.9, config_.min_relaxation);
    } else {
        // Poor progress or diverging - decrease significantly
        return std::max(config_.relaxation_factor * 0.5, config_.min_relaxation);
    }
}

void SteadyStateSolver::updatePressures(const Eigen::VectorXd& delta_p, double step_size,
                                       SolutionResults& results) {
    for (size_t i = 0; i < unknown_pressure_nodes_.size(); ++i) {
        const std::string& node_id = unknown_pressure_nodes_[i];
        double new_pressure = results.node_pressures[node_id] + step_size * delta_p(static_cast<Eigen::Index>(i));
        
        // Apply physical limits
        new_pressure = std::max(0.1e5, std::min(1000e5, new_pressure));  // 0.1-1000 bar
        
        results.node_pressures[node_id] = new_pressure;
        network_->nodes().at(node_id)->set_pressure(new_pressure);
    }
}

void SteadyStateSolver::evaluateResidual(Eigen::VectorXd& F, const SolutionResults& results) {
    for (size_t i = 0; i < unknown_pressure_nodes_.size(); ++i) {
        const std::string& node_id = unknown_pressure_nodes_[i];
        auto node = network_->nodes().at(node_id);
        
        double flow_balance = 0.0;
        
        // Add flows from connected pipes
        for (const auto& pipe_id : node_to_pipes_[node_id]) {
            auto pipe = network_->pipes().at(pipe_id);
            double flow = results.pipe_flow_rates.at(pipe_id);
            
            if (pipe->upstream()->id() == node_id) {
                flow_balance -= flow;
            } else {
                flow_balance += flow;
            }
        }
        
        // Add source/sink terms
        if (node->fixed_flow_rate() != 0.0) {
            flow_balance += node->fixed_flow_rate();
        }
        
        F(static_cast<Eigen::Index>(i)) = -flow_balance;
    }
}

bool SteadyStateSolver::checkConvergence(double current_residual, double prev_residual,
                                        const std::vector<double>& recent_residuals, int iteration) {
    // Absolute tolerance
    if (current_residual < config_.tolerance) {
        return true;
    }
    
    // Relative tolerance
    if (config_.check_relative_tolerance && iteration > 0) {
        double relative_change = std::abs(current_residual - prev_residual) / 
                               (prev_residual + 1e-20);
        if (relative_change < config_.relative_tolerance) {
            return true;
        }
    }
    
    // Stagnation check
    if (recent_residuals.size() >= config_.stagnation_check_window) {
        double max_recent = *std::max_element(recent_residuals.begin(), recent_residuals.end());
        double min_recent = *std::min_element(recent_residuals.begin(), recent_residuals.end());
        
        if (max_recent - min_recent < config_.stagnation_tolerance) {
            if (config_.verbose) {
                printf("Stagnation detected\n");
            }
            return true;
        }
    }
    
    return false;
}

void SteadyStateSolver::calculateFinalResults(SolutionResults& results) {
    // Update all node states
    for (const auto& [id, node] : network_->nodes()) {
        node->set_pressure(results.node_pressures[id]);
        node->set_temperature(results.node_temperatures[id]);
    }
    
    // Update all pipe states and calculate final quantities
    for (const auto& [id, pipe] : network_->pipes()) {
        pipe->set_flow_rate(results.pipe_flow_rates[id]);
        
        // Additional results already calculated in calculatePipeFlowProfessional
    }
    
    // Calculate summary statistics
    if (config_.verbose && network_->pipes().size() > 0) {
        double total_flow = 0.0;
        double total_pressure_drop = 0.0;
        
        for (const auto& [id, flow] : results.pipe_flow_rates) {
            total_flow += std::abs(flow);
        }
        
        for (const auto& [id, dp] : results.pipe_pressure_drops) {
            total_pressure_drop += std::abs(dp);
        }
        
        printf("\n? Network Summary:\n");
        printf("  Total flow: %.3f m³/s (%.0f m³/day)\n", 
               total_flow / network_->pipes().size(), 
               total_flow * 86400 / network_->pipes().size());
        printf("  Average pressure drop: %.2f bar\n", 
               total_pressure_drop / network_->pipes().size() / 1e5);
    }
}

// TransientSolver stub implementation
TransientSolver::TransientSolver(std::shared_ptr<Network> network, const FluidProperties& fluid)
    : Solver(network, fluid) {
}

SolutionResults TransientSolver::solve() {
    SolutionResults results;
    results.converged = false;
    results.convergence_reason = "Transient solver not yet implemented";
    return results;
}

} // namespace pipeline_sim
