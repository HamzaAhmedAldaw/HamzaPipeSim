#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>
#include <Eigen/IterativeLinearSolvers>
#include <map>
#include <string>
#include <memory>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>

// Define constants for Windows
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Include only the headers that work
#include "pipeline_sim/types.h"
#include "pipeline_sim/node.h"
#include "pipeline_sim/pipe.h"
#include "pipeline_sim/fluid_properties.h"
#include "pipeline_sim/network.h"

// Professional-grade solver implementation
namespace pipeline_sim {
    
    // Enhanced solver configuration with industry-standard parameters
    struct SolverConfig {
        double tolerance = 1e-6;
        int max_iterations = 100;
        double relaxation_factor = 1.0;
        bool verbose = false;
        
        // Advanced parameters for robust convergence
        bool use_line_search = true;
        double line_search_alpha = 1e-4;  // Armijo constant
        double line_search_beta = 0.5;     // Backtracking factor
        int max_line_search_iter = 20;
        
        bool use_adaptive_relaxation = true;
        double min_relaxation = 0.1;
        double max_relaxation = 1.0;
        
        // Trust region parameters
        bool use_trust_region = false;
        double trust_region_radius = 1e6;  // Pa
        
        // Jacobian calculation method
        enum JacobianMethod {
            FINITE_DIFFERENCE,
            ANALYTICAL
        };
        JacobianMethod jacobian_method = ANALYTICAL;
        double finite_diff_step = 1000.0;  // Pa
        
        // Linear solver options
        enum LinearSolver {
            LU_DECOMPOSITION,
            QR_DECOMPOSITION,
            ITERATIVE_BICGSTAB
        };
        LinearSolver linear_solver = LU_DECOMPOSITION;
        
        // Convergence criteria
        bool check_relative_tolerance = true;
        double relative_tolerance = 1e-8;
        int stagnation_check_window = 5;
        double stagnation_tolerance = 1e-10;
    };
    
    // Enhanced solution results with detailed diagnostics
    struct SolutionResults {
        bool converged = false;
        int iterations = 0;
        double residual = 0.0;
        double computation_time = 0.0;
        
        std::map<std::string, double> node_pressures;
        std::map<std::string, double> node_temperatures;
        std::map<std::string, double> pipe_flow_rates;
        std::map<std::string, double> pipe_pressure_drops;
        
        // Additional results
        std::map<std::string, double> pipe_velocities;
        std::map<std::string, double> pipe_reynolds_numbers;
        std::map<std::string, double> pipe_friction_factors;
        
        // Convergence history for analysis
        std::vector<double> residual_history;
        std::vector<double> step_size_history;
        
        // Diagnostics
        double jacobian_condition_number = 0.0;
        int jacobian_rank = 0;
        std::string convergence_reason = "";
    };
    
    // Professional pipeline solver with industry-standard algorithms
    class PipelineSolver {
    public:
        PipelineSolver(std::shared_ptr<Network> network, const FluidProperties& fluid) 
            : network_(network), fluid_(fluid) {
            setupSolver();
        }
        
        SolutionResults solve() {
            auto start_time = std::chrono::high_resolution_clock::now();
            SolutionResults results;
            
            if (!network_ || network_->nodes().empty() || network_->pipes().empty()) {
                results.converged = false;
                results.convergence_reason = "Invalid network";
                return results;
            }
            
            // Professional solver setup
            if (config.verbose) {
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
            
            for (int iter = 0; iter < config.max_iterations; ++iter) {
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
                    
                    if (config.verbose) {
                        printf("\n? Converged in %d iterations! Final residual: %.2e\n", iter + 1, current_residual);
                    }
                    break;
                }
                
                // Step 2: Solve linear system for pressure corrections
                Eigen::VectorXd delta_p;
                bool solve_success = solveLinearSystem(J, F, delta_p, results);
                
                if (!solve_success) {
                    if (config.verbose) {
                        printf("\n??  Linear solver failed at iteration %d\n", iter + 1);
                    }
                    results.convergence_reason = "Linear solver failure";
                    break;
                }
                
                // Step 3: Apply pressure corrections with line search or trust region
                double step_size = 1.0;
                if (config.use_line_search) {
                    step_size = performLineSearch(delta_p, F, current_residual, results);
                } else if (config.use_trust_region) {
                    step_size = applyTrustRegion(delta_p, current_residual);
                } else if (config.use_adaptive_relaxation) {
                    step_size = calculateAdaptiveRelaxation(current_residual, prev_residual, iter);
                } else {
                    step_size = config.relaxation_factor;
                }
                
                results.step_size_history.push_back(step_size);
                
                // Update pressures
                updatePressures(delta_p, step_size, results);
                
                // Update pipe flows with new pressures
                updatePipeFlows(results);
                
                // Verbose output
                if (config.verbose && (iter % 10 == 0 || iter < 5)) {
                    printf("Iter %3d: |F|=%.2e, ||?p||=%.2e bar, step=%.3f\n", 
                           iter + 1, current_residual, delta_p.norm()/1e5, step_size);
                }
                
                // Check for divergence
                if (current_residual > 1e10 || std::isnan(current_residual)) {
                    if (config.verbose) {
                        printf("\n? Solver diverged at iteration %d\n", iter + 1);
                    }
                    results.convergence_reason = "Divergence detected";
                    break;
                }
                
                prev_residual = current_residual;
                recent_residuals.push_back(current_residual);
                if (recent_residuals.size() > config.stagnation_check_window) {
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
            
            if (config.verbose) {
                printSolverSummary(results);
            }
            
            if (!converged && results.convergence_reason.empty()) {
                results.convergence_reason = "Maximum iterations reached";
            }
            
            return results;
        }
        
        SolverConfig config;
        
    private:
        std::shared_ptr<Network> network_;
        FluidProperties fluid_;
        
        // System indexing
        std::vector<std::string> unknown_pressure_nodes_;
        std::map<std::string, size_t> node_to_index_;
        std::map<std::string, std::vector<std::string>> node_to_pipes_;
        
        void setupSolver() {
            // Any additional setup
        }
        
        void printSolverHeader() {
            printf("\n================================================================================\n");
            printf("  PROFESSIONAL PIPELINE NETWORK SOLVER v2.0\n");
            printf("  Method: Newton-Raphson with %s Jacobian\n", 
                   config.jacobian_method == SolverConfig::ANALYTICAL ? "Analytical" : "Finite Difference");
            printf("  Linear Solver: %s\n", 
                   config.linear_solver == SolverConfig::LU_DECOMPOSITION ? "LU Decomposition" :
                   config.linear_solver == SolverConfig::QR_DECOMPOSITION ? "QR Decomposition" : "BiCGSTAB");
            printf("================================================================================\n");
        }
        
        void printSolverSummary(const SolutionResults& results) {
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
        
        void buildSystemIndexing() {
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
        
        bool validateNetwork() {
            // Check if we have at least one pressure boundary condition
            bool has_pressure_bc = false;
            for (const auto& [node_id, node] : network_->nodes()) {
                if (node->has_pressure_bc()) {
                    has_pressure_bc = true;
                    break;
                }
            }
            
            if (!has_pressure_bc) {
                if (config.verbose) {
                    printf("??  Error: Network has no pressure boundary conditions\n");
                }
                return false;
            }
            
            // Check network connectivity (simplified - should use graph algorithms)
            return true;
        }
        
        void initializeSolutionProfessional(SolutionResults& results) {
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
        
        void updatePipeFlows(SolutionResults& results) {
            for (const auto& [pipe_id, pipe] : network_->pipes()) {
                calculatePipeFlowProfessional(pipe, results);
            }
        }
        
        void calculatePipeFlowProfessional(const std::shared_ptr<Pipe>& pipe, 
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
        
        double calculateFrictionFactorColebrook(double Re, double D, double eps) {
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
        
        void assembleSystemOfEquations(Eigen::SparseMatrix<double>& J, 
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
                if (config.jacobian_method == SolverConfig::ANALYTICAL) {
                    calculateAnalyticalJacobian(static_cast<int>(i), node_id, triplets, results);
                } else {
                    calculateFiniteDifferenceJacobian(static_cast<int>(i), node_id, J, results, F);
                }
            }
            
            if (config.jacobian_method == SolverConfig::ANALYTICAL) {
                J.setFromTriplets(triplets.begin(), triplets.end());
            }
            
            // Make matrix more stable by adding small diagonal term
            for (size_t i = 0; i < n; ++i) {
                J.coeffRef(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(i)) += 1e-10;
            }
        }
        
        void calculateAnalyticalJacobian(int row, const std::string& node_id,
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
        
        void calculateFiniteDifferenceJacobian(int row, const std::string& node_id,
                                             Eigen::SparseMatrix<double>& J,
                                             const SolutionResults& results,
                                             const Eigen::VectorXd& F_current) {
            // Finite difference approximation of Jacobian
            double h = config.finite_diff_step;
            
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
        
        bool solveLinearSystem(const Eigen::SparseMatrix<double>& J,
                              const Eigen::VectorXd& F,
                              Eigen::VectorXd& delta_p,
                              SolutionResults& results) {
            try {
                if (config.linear_solver == SolverConfig::LU_DECOMPOSITION) {
                    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
                    solver.analyzePattern(J);
                    solver.factorize(J);
                    
                    if (solver.info() != Eigen::Success) {
                        return false;
                    }
                    
                    delta_p = solver.solve(F);
                    
                    // Calculate condition number estimate (not available in Eigen SparseLU)
                    // results.jacobian_condition_number = 1.0; // Placeholder
                    
                } else if (config.linear_solver == SolverConfig::QR_DECOMPOSITION) {
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
                if (config.verbose) {
                    printf("Linear solver exception: %s\n", e.what());
                }
                return false;
            }
        }
        
        double performLineSearch(const Eigen::VectorXd& delta_p,
                               const Eigen::VectorXd& F0,
                               double residual0,
                               SolutionResults& results) {
            double alpha = 1.0;
            const double c1 = config.line_search_alpha;  // Armijo constant
            const double rho = config.line_search_beta;  // Backtracking factor
            
            // Save original state
            std::map<std::string, double> original_pressures = results.node_pressures;
            
            // Expected decrease (directional derivative)
            double expected_decrease = -F0.dot(delta_p);
            
            if (expected_decrease >= 0) {
                // Not a descent direction
                return config.relaxation_factor;
            }
            
            // Line search loop
            for (int ls_iter = 0; ls_iter < config.max_line_search_iter; ++ls_iter) {
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
        
        double applyTrustRegion(Eigen::VectorXd& delta_p, double current_residual) {
            double step_norm = delta_p.norm();
            
            if (step_norm > config.trust_region_radius) {
                // Scale step to fit in trust region
                delta_p *= config.trust_region_radius / step_norm;
                return config.trust_region_radius / step_norm;
            }
            
            return 1.0;
        }
        
        double calculateAdaptiveRelaxation(double current_residual, 
                                         double prev_residual,
                                         int iteration) {
            double ratio = current_residual / prev_residual;
            
            if (ratio < 0.5) {
                // Excellent progress - increase relaxation
                return std::min(config.relaxation_factor * 1.2, config.max_relaxation);
            } else if (ratio < 0.9) {
                // Good progress - keep current
                return config.relaxation_factor;
            } else if (ratio < 1.1) {
                // Slow progress - decrease slightly
                return std::max(config.relaxation_factor * 0.9, config.min_relaxation);
            } else {
                // Poor progress or diverging - decrease significantly
                return std::max(config.relaxation_factor * 0.5, config.min_relaxation);
            }
        }
        
        void updatePressures(const Eigen::VectorXd& delta_p, double step_size,
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
        
        void evaluateResidual(Eigen::VectorXd& F, const SolutionResults& results) {
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
        
        bool checkConvergence(double current_residual, double prev_residual,
                            const std::vector<double>& recent_residuals, int iteration) {
            // Absolute tolerance
            if (current_residual < config.tolerance) {
                return true;
            }
            
            // Relative tolerance
            if (config.check_relative_tolerance && iteration > 0) {
                double relative_change = std::abs(current_residual - prev_residual) / 
                                       (prev_residual + 1e-20);
                if (relative_change < config.relative_tolerance) {
                    return true;
                }
            }
            
            // Stagnation check
            if (recent_residuals.size() >= config.stagnation_check_window) {
                double max_recent = *std::max_element(recent_residuals.begin(), recent_residuals.end());
                double min_recent = *std::min_element(recent_residuals.begin(), recent_residuals.end());
                
                if (max_recent - min_recent < config.stagnation_tolerance) {
                    if (config.verbose) {
                        printf("Stagnation detected\n");
                    }
                    return true;
                }
            }
            
            return false;
        }
        
        void calculateFinalResults(SolutionResults& results) {
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
            if (config.verbose && network_->pipes().size() > 0) {
                double total_flow = 0.0;
                double total_pressure_drop = 0.0;
                
                for (const auto& [id, flow] : results.pipe_flow_rates) {
                    total_flow += std::abs(flow);
                }
                
                for (const auto& [id, dp] : results.pipe_pressure_drops) {
                    total_pressure_drop += std::abs(dp);
                }
                
                printf("\n?? Network Summary:\n");
                printf("  Total flow: %.3f m³/s (%.0f m³/day)\n", 
                       total_flow / network_->pipes().size(), 
                       total_flow * 86400 / network_->pipes().size());
                printf("  Average pressure drop: %.2f bar\n", 
                       total_pressure_drop / network_->pipes().size() / 1e5);
            }
        }
    };
}

namespace py = pybind11;

PYBIND11_MODULE(pipeline_sim, m) {
    m.doc() = "Pipeline-Sim: Professional Pipeline Simulation (v2.0)";
    
    // NodeType enum
    py::enum_<pipeline_sim::NodeType>(m, "NodeType")
        .value("JUNCTION", pipeline_sim::NodeType::JUNCTION)
        .value("SOURCE", pipeline_sim::NodeType::SOURCE)
        .value("SINK", pipeline_sim::NodeType::SINK)
        .value("PUMP", pipeline_sim::NodeType::PUMP)
        .value("COMPRESSOR", pipeline_sim::NodeType::COMPRESSOR)
        .value("VALVE", pipeline_sim::NodeType::VALVE)
        .value("SEPARATOR", pipeline_sim::NodeType::SEPARATOR)
        .value("HEAT_EXCHANGER", pipeline_sim::NodeType::HEAT_EXCHANGER);
    
    // Node
    py::class_<pipeline_sim::Node, std::shared_ptr<pipeline_sim::Node>>(m, "Node")
        .def(py::init<const std::string&, pipeline_sim::NodeType>())
        .def("id", &pipeline_sim::Node::id)
        .def("type", &pipeline_sim::Node::type)
        .def("pressure", &pipeline_sim::Node::pressure)
        .def("set_pressure", &pipeline_sim::Node::set_pressure)
        .def("temperature", &pipeline_sim::Node::temperature)
        .def("set_temperature", &pipeline_sim::Node::set_temperature)
        .def("elevation", &pipeline_sim::Node::elevation)
        .def("set_elevation", &pipeline_sim::Node::set_elevation)
        .def("has_pressure_bc", &pipeline_sim::Node::has_pressure_bc)
        .def("set_pressure_bc", &pipeline_sim::Node::set_pressure_bc)
        .def("pressure_bc", &pipeline_sim::Node::pressure_bc)
        .def("fixed_flow_rate", &pipeline_sim::Node::fixed_flow_rate)
        .def("set_fixed_flow_rate", &pipeline_sim::Node::set_fixed_flow_rate)
        .def("set_flow_rate", &pipeline_sim::Node::set_fixed_flow_rate,
             "Set flow rate for node (alias for set_fixed_flow_rate)")
        .def("pump_speed", &pipeline_sim::Node::pump_speed)
        .def("set_pump_speed", &pipeline_sim::Node::set_pump_speed)
        .def("set_pump_curve", &pipeline_sim::Node::set_pump_curve)
        .def("compressor_ratio", &pipeline_sim::Node::compressor_ratio)
        .def("set_compressor_ratio", &pipeline_sim::Node::set_compressor_ratio)
        .def("__repr__", [](const pipeline_sim::Node& n) {
            return "<Node '" + n.id() + "' P=" + 
                   std::to_string(n.pressure()/1e5) + " bar, z=" +
                   std::to_string(n.elevation()) + " m>";
        });
    
    // Pipe
    py::class_<pipeline_sim::Pipe, std::shared_ptr<pipeline_sim::Pipe>>(m, "Pipe")
        .def(py::init<const std::string&, 
                      std::shared_ptr<pipeline_sim::Node>, 
                      std::shared_ptr<pipeline_sim::Node>, 
                      double, double>())
        .def("id", &pipeline_sim::Pipe::id)
        .def("upstream", &pipeline_sim::Pipe::upstream)
        .def("downstream", &pipeline_sim::Pipe::downstream)
        .def("length", &pipeline_sim::Pipe::length)
        .def("diameter", &pipeline_sim::Pipe::diameter)
        .def("roughness", &pipeline_sim::Pipe::roughness)
        .def("set_roughness", &pipeline_sim::Pipe::set_roughness)
        .def("inclination", &pipeline_sim::Pipe::inclination)
        .def("set_inclination", &pipeline_sim::Pipe::set_inclination)
        .def("area", &pipeline_sim::Pipe::area)
        .def("volume", &pipeline_sim::Pipe::volume)
        .def("flow_rate", &pipeline_sim::Pipe::flow_rate)
        .def("set_flow_rate", &pipeline_sim::Pipe::set_flow_rate)
        .def("velocity", &pipeline_sim::Pipe::velocity)
        .def("reynolds_number", [](const pipeline_sim::Pipe& p, double viscosity, double density) {
            return p.reynolds_number(viscosity, density);
        })
        .def("friction_factor", [](const pipeline_sim::Pipe& p, double reynolds) {
            return p.friction_factor(reynolds);
        })
        .def("__repr__", [](const pipeline_sim::Pipe& p) {
            return "<Pipe '" + p.id() + "' L=" + 
                   std::to_string(p.length()) + "m D=" + 
                   std::to_string(p.diameter()*39.37) + "\">";
        });
    
    // FluidProperties
    py::class_<pipeline_sim::FluidProperties>(m, "FluidProperties")
        .def(py::init<>())
        .def_readwrite("oil_density", &pipeline_sim::FluidProperties::oil_density)
        .def_readwrite("gas_density", &pipeline_sim::FluidProperties::gas_density)
        .def_readwrite("water_density", &pipeline_sim::FluidProperties::water_density)
        .def_readwrite("oil_viscosity", &pipeline_sim::FluidProperties::oil_viscosity)
        .def_readwrite("gas_viscosity", &pipeline_sim::FluidProperties::gas_viscosity)
        .def_readwrite("water_viscosity", &pipeline_sim::FluidProperties::water_viscosity)
        .def_readwrite("oil_fraction", &pipeline_sim::FluidProperties::oil_fraction)
        .def_readwrite("gas_fraction", &pipeline_sim::FluidProperties::gas_fraction)
        .def_readwrite("water_fraction", &pipeline_sim::FluidProperties::water_fraction)
        .def_readwrite("temperature", &pipeline_sim::FluidProperties::temperature)
        .def_readwrite("pressure", &pipeline_sim::FluidProperties::pressure)
        .def_readwrite("gas_oil_ratio", &pipeline_sim::FluidProperties::gas_oil_ratio)
        .def_readwrite("water_cut", &pipeline_sim::FluidProperties::water_cut)
        .def("mixture_density", &pipeline_sim::FluidProperties::mixture_density)
        .def("mixture_viscosity", &pipeline_sim::FluidProperties::mixture_viscosity)
        .def("is_multiphase", &pipeline_sim::FluidProperties::is_multiphase)
        .def("liquid_fraction", &pipeline_sim::FluidProperties::liquid_fraction);
    
    // Network
    py::class_<pipeline_sim::Network, std::shared_ptr<pipeline_sim::Network>>(m, "Network")
        .def(py::init<>())
        .def("add_node", &pipeline_sim::Network::add_node)
        .def("add_pipe", &pipeline_sim::Network::add_pipe)
        .def("get_node", &pipeline_sim::Network::get_node)
        .def("get_pipe", &pipeline_sim::Network::get_pipe)
        .def("nodes", &pipeline_sim::Network::nodes)
        .def("pipes", &pipeline_sim::Network::pipes)
        .def("set_pressure", 
             static_cast<void (pipeline_sim::Network::*)(const std::shared_ptr<pipeline_sim::Node>&, double)>(&pipeline_sim::Network::set_pressure))
        .def("set_flow_rate", 
             static_cast<void (pipeline_sim::Network::*)(const std::shared_ptr<pipeline_sim::Node>&, double)>(&pipeline_sim::Network::set_flow_rate))
        .def("node_count", &pipeline_sim::Network::node_count)
        .def("pipe_count", &pipeline_sim::Network::pipe_count)
        .def("is_valid", &pipeline_sim::Network::is_valid)
        .def("clear", &pipeline_sim::Network::clear)
        .def("pressure_specs", &pipeline_sim::Network::pressure_specs)
        .def("flow_specs", &pipeline_sim::Network::flow_specs)
        .def("load_from_json", &pipeline_sim::Network::load_from_json)
        .def("save_to_json", &pipeline_sim::Network::save_to_json);
    
    // SolverConfig
    py::class_<pipeline_sim::SolverConfig>(m, "SolverConfig")
        .def(py::init<>())
        .def_readwrite("tolerance", &pipeline_sim::SolverConfig::tolerance)
        .def_readwrite("max_iterations", &pipeline_sim::SolverConfig::max_iterations)
        .def_readwrite("relaxation_factor", &pipeline_sim::SolverConfig::relaxation_factor)
        .def_readwrite("verbose", &pipeline_sim::SolverConfig::verbose)
        .def_readwrite("use_line_search", &pipeline_sim::SolverConfig::use_line_search)
        .def_readwrite("line_search_alpha", &pipeline_sim::SolverConfig::line_search_alpha)
        .def_readwrite("line_search_beta", &pipeline_sim::SolverConfig::line_search_beta)
        .def_readwrite("max_line_search_iter", &pipeline_sim::SolverConfig::max_line_search_iter)
        .def_readwrite("use_adaptive_relaxation", &pipeline_sim::SolverConfig::use_adaptive_relaxation)
        .def_readwrite("min_relaxation", &pipeline_sim::SolverConfig::min_relaxation)
        .def_readwrite("max_relaxation", &pipeline_sim::SolverConfig::max_relaxation)
        .def_readwrite("use_trust_region", &pipeline_sim::SolverConfig::use_trust_region)
        .def_readwrite("trust_region_radius", &pipeline_sim::SolverConfig::trust_region_radius)
        .def_readwrite("check_relative_tolerance", &pipeline_sim::SolverConfig::check_relative_tolerance)
        .def_readwrite("relative_tolerance", &pipeline_sim::SolverConfig::relative_tolerance)
        .def_readwrite("stagnation_check_window", &pipeline_sim::SolverConfig::stagnation_check_window)
        .def_readwrite("stagnation_tolerance", &pipeline_sim::SolverConfig::stagnation_tolerance);
    
    // SolutionResults
    py::class_<pipeline_sim::SolutionResults>(m, "SolutionResults")
        .def(py::init<>())
        .def_readonly("converged", &pipeline_sim::SolutionResults::converged)
        .def_readonly("iterations", &pipeline_sim::SolutionResults::iterations)
        .def_readonly("residual", &pipeline_sim::SolutionResults::residual)
        .def_readonly("computation_time", &pipeline_sim::SolutionResults::computation_time)
        .def_readonly("node_pressures", &pipeline_sim::SolutionResults::node_pressures)
        .def_readonly("node_temperatures", &pipeline_sim::SolutionResults::node_temperatures)
        .def_readonly("pipe_flow_rates", &pipeline_sim::SolutionResults::pipe_flow_rates)
        .def_readonly("pipe_pressure_drops", &pipeline_sim::SolutionResults::pipe_pressure_drops)
        .def_readonly("pipe_velocities", &pipeline_sim::SolutionResults::pipe_velocities)
        .def_readonly("pipe_reynolds_numbers", &pipeline_sim::SolutionResults::pipe_reynolds_numbers)
        .def_readonly("pipe_friction_factors", &pipeline_sim::SolutionResults::pipe_friction_factors)
        .def_readonly("residual_history", &pipeline_sim::SolutionResults::residual_history)
        .def_readonly("step_size_history", &pipeline_sim::SolutionResults::step_size_history)
        .def_readonly("jacobian_condition_number", &pipeline_sim::SolutionResults::jacobian_condition_number)
        .def_readonly("jacobian_rank", &pipeline_sim::SolutionResults::jacobian_rank)
        .def_readonly("convergence_reason", &pipeline_sim::SolutionResults::convergence_reason)
        .def("__repr__", [](const pipeline_sim::SolutionResults& r) {
            return "<SolutionResults converged=" + std::string(r.converged ? "True" : "False") + 
                   " iterations=" + std::to_string(r.iterations) + 
                   " residual=" + std::to_string(r.residual) + ">";
        });
    
    // PipelineSolver exposed as SteadyStateSolver
    py::class_<pipeline_sim::PipelineSolver>(m, "SteadyStateSolver")
        .def(py::init<std::shared_ptr<pipeline_sim::Network>, const pipeline_sim::FluidProperties&>())
        .def("solve", &pipeline_sim::PipelineSolver::solve)
        .def_readwrite("config", &pipeline_sim::PipelineSolver::config);
    
    // Module attributes
    m.attr("__version__") = "2.0.0";
    m.attr("__author__") = "Pipeline-Sim Professional Team";
    
    // Module-level functions
    m.def("create_example_network", []() {
        auto network = std::make_shared<pipeline_sim::Network>();
        auto inlet = network->add_node("INLET", pipeline_sim::NodeType::SOURCE);
        auto outlet = network->add_node("OUTLET", pipeline_sim::NodeType::SINK);
        inlet->set_elevation(0.0);
        outlet->set_elevation(0.0);
        auto pipe = network->add_pipe("PIPE1", inlet, outlet, 1000.0, 0.3048);
        pipe->set_roughness(0.000045);
        network->set_pressure(inlet, 70e5);  // 70 bar
        network->set_pressure(outlet, 69e5);  // 69 bar
        return network;
    }, "Create a simple example network for testing");
    
    m.def("create_example_fluid", []() {
        pipeline_sim::FluidProperties fluid;
        fluid.oil_density = 850;
        fluid.oil_viscosity = 0.002;
        fluid.oil_fraction = 1.0;
        fluid.gas_fraction = 0.0;
        fluid.water_fraction = 0.0;
        return fluid;
    }, "Create example fluid properties (light oil)");
    
    m.def("get_version", []() {
        return std::string("2.0.0");
    }, "Get Pipeline-Sim version");
}
