#include "pipeline_sim/solver.h"
#include <Eigen/IterativeLinearSolvers>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace pipeline_sim {

// Constants
constexpr Real PI = 3.14159265358979323846;
constexpr Real LAMINAR_REYNOLDS = 2000.0;
constexpr Real TURBULENT_REYNOLDS = 4000.0;
constexpr Real MIN_VELOCITY = 0.001;  // m/s
constexpr Real MIN_REYNOLDS = 10.0;

//=============================================================================
// Base Solver Implementation
//=============================================================================

Solver::Solver(Ptr<Network> network, const FluidProperties& fluid)
    : network_(network), fluid_(fluid) {
}

Real Solver::calculate_condition_number(const SparseMatrix& A) {
    // Estimate condition number using power method
    Eigen::VectorXd v = Eigen::VectorXd::Random(A.cols());
    v.normalize();
    
    Real lambda_max = 0.0;
    for (int i = 0; i < 20; ++i) {
        Eigen::VectorXd Av = A * v;
        lambda_max = Av.norm();
        v = Av / lambda_max;
    }
    
    // For minimum eigenvalue, use inverse power method (approximate)
    Real lambda_min = 1.0 / lambda_max;  // Rough estimate
    
    return lambda_max / lambda_min;
}

Real Solver::check_mass_balance() {
    Real total_imbalance = 0.0;
    
    for (const auto& [node_id, node] : network_->nodes()) {
        Real inflow = 0.0;
        Real outflow = 0.0;
        
        // Sum flows at node
        for (const auto& pipe : network_->get_upstream_pipes(node)) {
            outflow += std::abs(pipe->flow_rate());
        }
        
        for (const auto& pipe : network_->get_downstream_pipes(node)) {
            inflow += std::abs(pipe->flow_rate());
        }
        
        // Check external flows
        if (network_->flow_specs().count(node_id) > 0) {
            Real external = network_->flow_specs().at(node_id);
            if (external > 0) inflow += external;
            else outflow += std::abs(external);
        }
        
        Real imbalance = std::abs(inflow - outflow);
        total_imbalance += imbalance;
    }
    
    return total_imbalance;
}

//=============================================================================
// Global Newton-Raphson Solver Implementation
//=============================================================================

SolutionResults GlobalNewtonRaphsonSolver::solve() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    SolutionResults results;
    results.converged = false;
    
    if (config_.verbose) {
        std::cout << "\n=== Global Newton-Raphson Solver ===" << std::endl;
        std::cout << "Network: " << network_->nodes().size() << " nodes, " 
                  << network_->pipes().size() << " pipes" << std::endl;
    }
    
    // Initialize system
    initialize_system();
    
    // Main iteration loop
    Vector x(num_unknowns_);
    Vector x_old(num_unknowns_);
    Vector dx(num_unknowns_);
    
    // Initial guess
    x.head(num_nodes_ - num_fixed_nodes_) = pressures_;
    x.tail(num_pipes_) = flows_;
    
    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        x_old = x;
        
        // Build Jacobian matrix and residual vector
        SparseMatrix A(num_unknowns_, num_unknowns_);
        Vector b(num_unknowns_);
        
        build_system_matrix(A, b);
        
        // Calculate residual
        Vector residual = A * x - b;
        results.residual = residual.norm();
        
        if (config_.verbose && iter % 10 == 0) {
            std::cout << "Iteration " << iter << ": residual = " 
                     << results.residual << std::endl;
        }
        
        // Check convergence
        if (check_convergence(residual)) {
            results.converged = true;
            results.iterations = iter + 1;
            break;
        }
        
        // Solve for correction
        bool solve_success = false;
        
        // Try different solvers
        if (!solve_success) {
            solve_success = solve_with_LU(A, -residual, dx);
        }
        
        if (!solve_success) {
            if (config_.verbose) {
                std::cout << "LU decomposition failed, trying QR..." << std::endl;
            }
            solve_success = solve_with_QR(A, -residual, dx);
        }
        
        if (!solve_success) {
            if (config_.verbose) {
                std::cout << "Direct solvers failed, trying iterative..." << std::endl;
            }
            solve_success = solve_with_iterative(A, -residual, dx);
        }
        
        if (!solve_success) {
            std::cerr << "All solution methods failed!" << std::endl;
            break;
        }
        
        // Apply damping if needed
        if (config_.use_damping) {
            Real damping = calculate_damping_factor(dx);
            apply_damping(dx, damping);
        }
        
        // Update solution
        x = x + config_.relaxation_factor * dx;
        
        // Update internal state
        pressures_ = x.head(num_nodes_ - num_fixed_nodes_);
        flows_ = x.tail(num_pipes_);
        update_solution(x);
        
        // Calculate condition number periodically
        if (iter % 20 == 0) {
            results.condition_number = calculate_condition_number(A);
            if (config_.verbose) {
                std::cout << "Condition number: " << results.condition_number << std::endl;
            }
        }
    }
    
    // Final update
    update_solution(x);
    
    // Store results
    for (const auto& [id, node] : network_->nodes()) {
        results.node_pressures[id] = node->pressure();
        results.node_temperatures[id] = node->temperature();
    }
    
    for (const auto& [id, pipe] : network_->pipes()) {
        results.pipe_flow_rates[id] = pipe->flow_rate();
        results.pipe_pressure_drops[id] = calculate_head_loss(pipe, pipe->flow_rate()) * 
                                         fluid_.mixture_density() * constants::GRAVITY;
        results.pipe_velocities[id] = pipe->velocity();
    }
    
    // Check mass balance
    if (config_.check_mass_balance) {
        results.mass_imbalance = check_mass_balance();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    results.computation_time = std::chrono::duration<Real>(end_time - start_time).count();
    
    if (config_.verbose) {
        std::cout << "\nSolver finished:" << std::endl;
        std::cout << "  Converged: " << (results.converged ? "Yes" : "No") << std::endl;
        std::cout << "  Iterations: " << results.iterations << std::endl;
        std::cout << "  Final residual: " << results.residual << std::endl;
        std::cout << "  Mass imbalance: " << results.mass_imbalance << std::endl;
        std::cout << "  Time: " << results.computation_time << " s" << std::endl;
    }
    
    return results;
}

void GlobalNewtonRaphsonSolver::initialize_system() {
    // Count fixed and free nodes
    num_fixed_nodes_ = 0;
    for (const auto& [id, pressure] : network_->pressure_specs()) {
        num_fixed_nodes_++;
    }
    
    // Order nodes and pipes
    node_order_.clear();
    pipe_order_.clear();
    node_to_index_.clear();
    pipe_to_index_.clear();
    
    // First add free nodes
    int idx = 0;
    for (const auto& [id, node] : network_->nodes()) {
        if (network_->pressure_specs().count(id) == 0) {
            node_order_.push_back(id);
            node_to_index_[id] = idx++;
        }
    }
    
    // Then add fixed nodes
    for (const auto& [id, pressure] : network_->pressure_specs()) {
        node_order_.push_back(id);
        node_to_index_[id] = idx++;
    }
    
    // Order pipes
    idx = 0;
    for (const auto& [id, pipe] : network_->pipes()) {
        pipe_order_.push_back(id);
        pipe_to_index_[id] = idx++;
    }
    
    num_nodes_ = static_cast<int>(node_order_.size());  // Fix: explicit cast to int
    num_pipes_ = static_cast<int>(pipe_order_.size());  // Fix: explicit cast to int
    num_unknowns_ = (num_nodes_ - num_fixed_nodes_) + num_pipes_;
    
    // Initialize state vectors
    pressures_.resize(num_nodes_ - num_fixed_nodes_);
    flows_.resize(num_pipes_);
    
    // Initial pressure guess
    Real avg_pressure = 101325.0;  // 1 atm default
    if (!network_->pressure_specs().empty()) {
        avg_pressure = 0.0;
        for (const auto& [id, p] : network_->pressure_specs()) {
            avg_pressure += p;
        }
        avg_pressure /= network_->pressure_specs().size();
    }
    
    pressures_.fill(avg_pressure);
    
    // Initial flow guess based on demands
    flows_.fill(0.001);  // Small positive flow
}

void GlobalNewtonRaphsonSolver::build_system_matrix(SparseMatrix& A, Vector& b) {
    std::vector<Eigen::Triplet<Real>> triplets;
    triplets.reserve(num_unknowns_ * 10);  // Estimate
    
    b.setZero();
    int eq_idx = 0;
    
    // 1. Continuity equations (mass balance at nodes)
    add_continuity_equations(triplets, b, eq_idx);
    
    // 2. Energy equations (head loss in pipes)
    add_energy_equations(triplets, b, eq_idx);
    
    // 3. Fixed grade equations (for nodes with specified pressure)
    add_fixed_grade_equations(triplets, b, eq_idx);
    
    // Build sparse matrix
    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();
}

void GlobalNewtonRaphsonSolver::add_continuity_equations(
    std::vector<Eigen::Triplet<Real>>& triplets, Vector& b, int& eq_idx) {
    
    // For each free node, sum of flows = external demand
    for (int i = 0; i < num_nodes_ - num_fixed_nodes_; ++i) {
        const std::string& node_id = node_order_[i];
        auto node = network_->get_node(node_id);
        
        // Add coefficients for connected pipes
        for (const auto& pipe : network_->get_upstream_pipes(node)) {
            int pipe_idx = pipe_to_index_[pipe->id()];
            int var_idx = (num_nodes_ - num_fixed_nodes_) + pipe_idx;
            triplets.push_back(Eigen::Triplet<Real>(eq_idx, var_idx, -1.0));
        }
        
        for (const auto& pipe : network_->get_downstream_pipes(node)) {
            int pipe_idx = pipe_to_index_[pipe->id()];
            int var_idx = (num_nodes_ - num_fixed_nodes_) + pipe_idx;
            triplets.push_back(Eigen::Triplet<Real>(eq_idx, var_idx, 1.0));
        }
        
        // RHS: external demand (negative for withdrawal)
        Real demand = 0.0;
        if (network_->flow_specs().count(node_id) > 0) {
            demand = -network_->flow_specs().at(node_id);
        }
        b(eq_idx) = demand;
        
        eq_idx++;
    }
}

void GlobalNewtonRaphsonSolver::add_energy_equations(
    std::vector<Eigen::Triplet<Real>>& triplets, Vector& b, int& eq_idx) {
    
    // For each pipe: H_upstream - H_downstream - h_loss = 0
    for (int i = 0; i < num_pipes_; ++i) {
        const std::string& pipe_id = pipe_order_[i];
        auto pipe = network_->get_pipe(pipe_id);
        
        std::string up_id = pipe->upstream()->id();
        std::string down_id = pipe->downstream()->id();
        
        // Get node indices
        int up_idx = node_to_index_[up_id];
        int down_idx = node_to_index_[down_id];
        
        // Coefficients for pressure terms (if node is free)
        if (up_idx < num_nodes_ - num_fixed_nodes_) {
            // Upstream pressure coefficient: 1/(?g)
            Real coeff = 1.0 / (fluid_.mixture_density() * constants::GRAVITY);
            triplets.push_back(Eigen::Triplet<Real>(eq_idx, up_idx, coeff));
        }
        
        if (down_idx < num_nodes_ - num_fixed_nodes_) {
            // Downstream pressure coefficient: -1/(?g)
            Real coeff = -1.0 / (fluid_.mixture_density() * constants::GRAVITY);
            triplets.push_back(Eigen::Triplet<Real>(eq_idx, down_idx, coeff));
        }
        
        // Head loss term
        Real flow = flows_(i);
        Real h_loss = calculate_head_loss(pipe, flow);
        Real dh_dq = calculate_head_loss_derivative(pipe, flow);
        
        // Flow coefficient
        int flow_var_idx = (num_nodes_ - num_fixed_nodes_) + i;
        triplets.push_back(Eigen::Triplet<Real>(eq_idx, flow_var_idx, -dh_dq));
        
        // RHS
        Real z_up = pipe->upstream()->elevation();
        Real z_down = pipe->downstream()->elevation();
        Real dz = z_up - z_down;
        
        // Add fixed pressures if any
        Real fixed_head = 0.0;
        if (network_->pressure_specs().count(up_id) > 0) {
            Real p_up = network_->pressure_specs().at(up_id);
            fixed_head += p_up / (fluid_.mixture_density() * constants::GRAVITY);
        }
        if (network_->pressure_specs().count(down_id) > 0) {
            Real p_down = network_->pressure_specs().at(down_id);
            fixed_head -= p_down / (fluid_.mixture_density() * constants::GRAVITY);
        }
        
        b(eq_idx) = -dz - fixed_head + h_loss - dh_dq * flow;
        
        eq_idx++;
    }
}

void GlobalNewtonRaphsonSolver::add_fixed_grade_equations(
    std::vector<Eigen::Triplet<Real>>& triplets, Vector& b, int& eq_idx) {
    
    // This is handled implicitly by ordering - fixed nodes are not unknowns
    // No additional equations needed
}

Real GlobalNewtonRaphsonSolver::calculate_head_loss(const Ptr<Pipe>& pipe, Real flow) {
    if (std::abs(flow) < MIN_VELOCITY * pipe->area()) {
        // Very low flow - use linear approximation
        return 0.0;
    }
    
    Real velocity = flow / pipe->area();
    Real reynolds = pipe->reynolds_number(fluid_.mixture_viscosity(), fluid_.mixture_density());
    
    // Ensure positive Reynolds
    reynolds = std::max(reynolds, MIN_REYNOLDS);
    
    Real friction = calculate_friction_factor(reynolds, pipe->roughness() / pipe->diameter());
    
    // Darcy-Weisbach equation: h_L = f * (L/D) * (v²/2g)
    Real h_loss = friction * pipe->length() / pipe->diameter() * 
                  velocity * std::abs(velocity) / (2.0 * constants::GRAVITY);
    
    return h_loss;
}

Real GlobalNewtonRaphsonSolver::calculate_head_loss_derivative(const Ptr<Pipe>& pipe, Real flow) {
    if (std::abs(flow) < MIN_VELOCITY * pipe->area()) {
        // Linear region
        Real K = 8.0 * fluid_.mixture_viscosity() * pipe->length() / 
                 (constants::GRAVITY * fluid_.mixture_density() * 
                  PI * std::pow(pipe->diameter(), 4));  // Fix: Use PI instead of M_PI
        return K;
    }
    
    Real area = pipe->area();
    Real velocity = flow / area;
    Real reynolds = std::abs(flow) * pipe->diameter() / (area * fluid_.mixture_viscosity());
    reynolds = std::max(reynolds, MIN_REYNOLDS);
    
    Real friction = calculate_friction_factor(reynolds, pipe->roughness() / pipe->diameter());
    
    // For turbulent flow: dh/dQ ˜ 2 * h_loss / Q
    Real dh_dq = 2.0 * friction * pipe->length() * std::abs(velocity) / 
                 (pipe->diameter() * area * 2.0 * constants::GRAVITY);
    
    return dh_dq;
}

Real GlobalNewtonRaphsonSolver::calculate_friction_factor(Real reynolds, Real relative_roughness) {
    if (reynolds < LAMINAR_REYNOLDS) {
        // Laminar flow: f = 64/Re
        return 64.0 / reynolds;
    }
    else if (reynolds < TURBULENT_REYNOLDS) {
        // Transition region - interpolate
        Real f_lam = 64.0 / LAMINAR_REYNOLDS;
        Real f_turb = 0.25 / std::pow(std::log10(relative_roughness / 3.7 + 5.74 / std::pow(TURBULENT_REYNOLDS, 0.9)), 2);
        Real t = (reynolds - LAMINAR_REYNOLDS) / (TURBULENT_REYNOLDS - LAMINAR_REYNOLDS);
        return f_lam * (1 - t) + f_turb * t;
    }
    else {
        // Turbulent flow - Swamee-Jain equation
        Real a = -2.0 * std::log10(relative_roughness / 3.7 + 5.74 / std::pow(reynolds, 0.9));
        return 0.25 / (a * a);
    }
}

bool GlobalNewtonRaphsonSolver::solve_with_LU(const SparseMatrix& A, const Vector& b, Vector& x) {
    try {
        Eigen::SparseLU<SparseMatrix> solver;
        solver.compute(A);
        
        if (solver.info() != Eigen::Success) {
            return false;
        }
        
        x = solver.solve(b);
        return solver.info() == Eigen::Success;
    }
    catch (...) {
        return false;
    }
}

bool GlobalNewtonRaphsonSolver::solve_with_QR(const SparseMatrix& A, const Vector& b, Vector& x) {
    try {
        Eigen::SparseQR<SparseMatrix, Eigen::COLAMDOrdering<int>> solver;
        solver.compute(A);
        
        if (solver.info() != Eigen::Success) {
            return false;
        }
        
        x = solver.solve(b);
        return solver.info() == Eigen::Success;
    }
    catch (...) {
        return false;
    }
}

bool GlobalNewtonRaphsonSolver::solve_with_iterative(const SparseMatrix& A, const Vector& b, Vector& x) {
    try {
        Eigen::BiCGSTAB<SparseMatrix> solver;
        solver.setMaxIterations(1000);
        solver.setTolerance(1e-8);
        solver.compute(A);
        
        if (solver.info() != Eigen::Success) {
            return false;
        }
        
        x = solver.solve(b);
        return solver.info() == Eigen::Success;
    }
    catch (...) {
        return false;
    }
}

Real GlobalNewtonRaphsonSolver::calculate_damping_factor(const Vector& dx) {
    // Limit maximum change in variables
    Real max_dp = 0.0;  // Maximum pressure change
    Real max_dq = 0.0;  // Maximum flow change
    
    for (int i = 0; i < num_nodes_ - num_fixed_nodes_; ++i) {
        max_dp = std::max(max_dp, std::abs(dx(i)));
    }
    
    for (int i = 0; i < num_pipes_; ++i) {
        int idx = (num_nodes_ - num_fixed_nodes_) + i;
        max_dq = std::max(max_dq, std::abs(dx(idx)));
    }
    
    // Limit pressure change to 10% of average pressure
    Real avg_p = pressures_.mean();
    Real damping_p = 1.0;
    if (max_dp > 0.1 * avg_p) {
        damping_p = 0.1 * avg_p / max_dp;
    }
    
    // Limit flow change to 50% of current flow
    Real damping_q = 1.0;
    Real max_flow = flows_.cwiseAbs().maxCoeff();
    if (max_dq > 0.5 * max_flow) {
        damping_q = 0.5 * max_flow / max_dq;
    }
    
    return std::max(config_.min_damping, std::min({damping_p, damping_q, config_.max_damping}));
}

void GlobalNewtonRaphsonSolver::apply_damping(Vector& dx, Real damping) {
    dx *= damping;
}

void GlobalNewtonRaphsonSolver::update_solution(const Vector& x) {
    // Update node pressures
    for (int i = 0; i < num_nodes_ - num_fixed_nodes_; ++i) {
        const std::string& node_id = node_order_[i];
        auto node = network_->get_node(node_id);
        node->set_pressure(x(i));
    }
    
    // Fixed node pressures
    for (const auto& [id, pressure] : network_->pressure_specs()) {
        auto node = network_->get_node(id);
        node->set_pressure(pressure);
    }
    
    // Update pipe flows
    for (int i = 0; i < num_pipes_; ++i) {
        const std::string& pipe_id = pipe_order_[i];
        auto pipe = network_->get_pipe(pipe_id);
        int idx = (num_nodes_ - num_fixed_nodes_) + i;
        pipe->set_flow_rate(x(idx));
    }
}

bool GlobalNewtonRaphsonSolver::check_convergence(const Vector& residual) {
    // Check both absolute and relative convergence
    Real abs_norm = residual.norm();
    Real rel_norm = abs_norm / std::max(1.0, pressures_.norm() + flows_.norm());
    
    return (abs_norm < config_.tolerance) || (rel_norm < config_.tolerance * 0.01);
}

//=============================================================================
// SolutionResults Implementation
//=============================================================================

Real SolutionResults::pressure_drop(const Ptr<Pipe>& pipe) const {
    auto it = pipe_pressure_drops.find(pipe->id());
    return (it != pipe_pressure_drops.end()) ? it->second : 0.0;
}

Real SolutionResults::outlet_pressure(const Ptr<Pipe>& pipe) const {
    auto it = node_pressures.find(pipe->downstream()->id());
    return (it != node_pressures.end()) ? it->second : 0.0;
}

} // namespace pipeline_sim
