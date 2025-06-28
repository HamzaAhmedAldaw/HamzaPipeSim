// ===== transient_solver.cpp =====
#include "pipeline_sim/transient_solver.h"
#include <Eigen/SparseLU>
#include <Eigen/IterativeLinearSolvers>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>

// Define M_PI for Windows compatibility
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace pipeline_sim {

// ===== TransientSolver Implementation =====
TransientSolver::TransientSolver(Ptr<Network> network, const FluidProperties& fluid)
    : Solver(network, fluid) {
    // Initialize steady-state solver for initial conditions
    steady_solver_ = std::make_unique<SteadyStateSolver>(network, fluid);
    
    // Set default transient config
    transient_config_ = TransientSolverConfig();
    config_ = transient_config_;  // Update base config
}

SolutionResults TransientSolver::solve() {
    // Delegate to transient solver
    return solve_transient();
}

TransientResults TransientSolver::solve_transient() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    TransientResults results;
    results.converged = false;
    results.iterations = 0;
    
    // First solve steady state for initial conditions
    std::cout << "\n=== Solving initial steady state ===" << std::endl;
    steady_solver_->set_config(config_);
    auto steady_results = steady_solver_->solve();
    
    if (!steady_results.converged) {
        std::cerr << "ERROR: Failed to obtain initial steady state solution" << std::endl;
        return results;
    }
    
    // Initialize solution vectors
    size_t num_nodes = network_->nodes().size();
    size_t num_pipes = network_->pipes().size();
    
    current_pressures_.resize(num_nodes);
    current_flows_.resize(num_pipes);
    previous_pressures_.resize(num_nodes);
    previous_flows_.resize(num_pipes);
    
    // Copy steady state solution
    size_t idx = 0;
    for (const auto& [id, node] : network_->nodes()) {
        current_pressures_(idx) = node->pressure();
        previous_pressures_(idx) = node->pressure();
        idx++;
    }
    
    idx = 0;
    for (const auto& [id, pipe] : network_->pipes()) {
        current_flows_(idx) = pipe->flow_rate();
        previous_flows_(idx) = pipe->flow_rate();
        idx++;
    }
    
    // Initialize time history storage
    initialize_solution_storage();
    store_solution_snapshot(0.0);
    
    // Calculate wave speeds if needed
    if (transient_config_.calculate_wave_speed) {
        for (const auto& [id, pipe] : network_->pipes()) {
            Real wave_speed = calculate_wave_speed_wood(pipe);
            // Store wave speed in pipe properties (would need to add this property)
            if (transient_config_.verbose) {
                std::cout << "Pipe " << id << " wave speed: " << wave_speed << " m/s" << std::endl;
            }
        }
    }
    
    // Main time stepping loop
    Real current_time = 0.0;
    Real last_output_time = 0.0;
    int total_steps = 0;
    
    std::cout << "\n=== Starting transient simulation ===" << std::endl;
    std::cout << "Total time: " << transient_config_.total_time << " s" << std::endl;
    std::cout << "Initial time step: " << transient_config_.time_step << " s" << std::endl;
    
    while (current_time < transient_config_.total_time) {
        // Calculate time step
        Real dt = transient_config_.time_step;
        if (transient_config_.adaptive_time_stepping) {
            dt = calculate_stable_time_step(current_time);
        }
        
        // Ensure we don't exceed total time
        if (current_time + dt > transient_config_.total_time) {
            dt = transient_config_.total_time - current_time;
        }
        
        // Apply transient events
        apply_events(current_time);
        
        // Advance solution
        bool step_success = advance_time_step(dt, current_time);
        
        if (!step_success) {
            if (transient_config_.adaptive_time_stepping && dt > transient_config_.min_time_step) {
                // Reduce time step and retry
                dt *= 0.5;
                continue;
            } else {
                std::cerr << "ERROR: Time step failed at t = " << current_time << std::endl;
                break;
            }
        }
        
        current_time += dt;
        total_steps++;
        
        // Store solution if needed
        if (current_time - last_output_time >= transient_config_.output_interval) {
            store_solution_snapshot(current_time);
            last_output_time = current_time;
            
            if (transient_config_.verbose) {
                std::cout << "Time = " << std::fixed << std::setprecision(3) << current_time 
                         << " s, dt = " << std::scientific << std::setprecision(3) << dt 
                         << " s, step = " << total_steps << std::endl;
            }
        }
        
        // Update previous solution
        previous_pressures_ = current_pressures_;
        previous_flows_ = current_flows_;
    }
    
    // Final analysis
    analyze_water_hammer();
    
    // Prepare results
    results.converged = true;
    results.iterations = total_steps;
    results.time_history = history_;
    
    // Copy final state
    idx = 0;
    for (const auto& [id, node] : network_->nodes()) {
        results.node_pressures[id] = current_pressures_(idx);
        idx++;
    }
    
    idx = 0;
    for (const auto& [id, pipe] : network_->pipes()) {
        results.pipe_flow_rates[id] = current_flows_(idx);
        idx++;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    results.computation_time = std::chrono::duration<Real>(end_time - start_time).count();
    
    std::cout << "\n=== Transient simulation completed ===" << std::endl;
    std::cout << "Total steps: " << total_steps << std::endl;
    std::cout << "Max pressure surge: " << results.max_pressure_surge << " Pa at " 
              << results.max_surge_location << " (t = " << results.max_surge_time << " s)" << std::endl;
    std::cout << "Computation time: " << results.computation_time << " s" << std::endl;
    
    return results;
}

bool TransientSolver::advance_time_step(Real dt, Real current_time) {
    // Choose integration scheme
    switch (transient_config_.integration_scheme) {
        case TimeIntegrationScheme::EXPLICIT_EULER:
            explicit_euler_step(dt);
            break;
        case TimeIntegrationScheme::IMPLICIT_EULER:
            implicit_euler_step(dt);
            break;
        case TimeIntegrationScheme::CRANK_NICOLSON:
            crank_nicolson_step(dt);
            break;
        case TimeIntegrationScheme::RUNGE_KUTTA_4:
            runge_kutta_4_step(dt);
            break;
    }
    
    // Check stability
    if (!check_courant_condition(dt) || !check_diffusion_condition(dt)) {
        if (transient_config_.verbose) {
            std::cout << "WARNING: Stability condition violated at t = " << current_time << std::endl;
        }
        return false;
    }
    
    // Update network state
    size_t idx = 0;
    for (const auto& [id, node] : network_->nodes()) {
        node->set_pressure(current_pressures_(idx));
        idx++;
    }
    
    idx = 0;
    for (const auto& [id, pipe] : network_->pipes()) {
        pipe->set_flow_rate(current_flows_(idx));
        pipe->set_velocity(current_flows_(idx) / pipe->area());
        idx++;
    }
    
    return true;
}

Real TransientSolver::calculate_stable_time_step(Real current_time) {
    Real dt = transient_config_.max_time_step;
    
    // Courant condition for each pipe
    for (const auto& [id, pipe] : network_->pipes()) {
        Real wave_speed = transient_config_.wave_speed;
        if (transient_config_.calculate_wave_speed) {
            wave_speed = calculate_wave_speed_wood(pipe);
        }
        
        Real velocity = std::abs(pipe->velocity());
        Real dx = pipe->length();
        
        Real courant_dt = transient_config_.courant_number * dx / (wave_speed + velocity);
        dt = std::min(dt, courant_dt);
    }
    
    // Diffusion condition (if viscous effects are significant)
    for (const auto& [id, pipe] : network_->pipes()) {
        Real nu = fluid_.mixture_viscosity() / fluid_.mixture_density();
        Real dx = pipe->length();
        
        Real diffusion_dt = transient_config_.diffusion_number * dx * dx / nu;
        dt = std::min(dt, diffusion_dt);
    }
    
    // Limit by event timing
    for (const auto& event : events_) {
        if (current_time < event.start_time && event.start_time < current_time + dt) {
            dt = event.start_time - current_time;
        }
    }
    
    return std::max(dt, transient_config_.min_time_step);
}

void TransientSolver::crank_nicolson_step(Real dt) {
    // Crank-Nicolson: theta = 0.5
    Real theta = 0.5;
    
    size_t n = current_pressures_.size() + current_flows_.size();
    SparseMatrix A(n, n);
    Vector b(n);
    Vector x(n);
    
    // Build system
    build_transient_system(A, b, dt, theta);
    
    // Solve
    Eigen::SparseLU<SparseMatrix> solver;
    solver.compute(A);
    
    if (solver.info() != Eigen::Success) {
        std::cerr << "ERROR: Matrix decomposition failed in transient solver" << std::endl;
        return;
    }
    
    x = solver.solve(b);
    
    // Extract solution
    for (int i = 0; i < current_pressures_.size(); ++i) {
        current_pressures_(i) = x(i);
    }
    
    for (int i = 0; i < current_flows_.size(); ++i) {
        current_flows_(i) = x(current_pressures_.size() + i);
    }
}

void TransientSolver::build_transient_system(SparseMatrix& A, Vector& b, Real dt, Real theta) {
    // This is a placeholder - actual implementation would build the full
    // transient system including mass matrix, stiffness matrix, etc.
    // For now, using simplified approach
    
    std::vector<Eigen::Triplet<Real>> triplets;
    
    size_t n_nodes = network_->nodes().size();
    size_t n_pipes = network_->pipes().size();
    
    // Simple explicit update for demonstration
    // In practice, this would be much more complex
    
    // Identity matrix for now
    for (size_t i = 0; i < n_nodes + n_pipes; ++i) {
        triplets.push_back(Eigen::Triplet<Real>(static_cast<int>(i), static_cast<int>(i), 1.0));
    }
    
    A.setFromTriplets(triplets.begin(), triplets.end());
    
    // Right hand side
    b.setZero();
    for (int i = 0; i < current_pressures_.size(); ++i) {
        b(i) = current_pressures_(i);
    }
    for (int i = 0; i < current_flows_.size(); ++i) {
        b(current_pressures_.size() + i) = current_flows_(i);
    }
}

Real TransientSolver::calculate_wave_speed_wood(const Ptr<Pipe>& pipe) {
    // Wood's formula for wave speed in pipes
    Real K = 2.14e9;  // Bulk modulus of water (Pa)
    Real rho = fluid_.mixture_density();
    Real E = 2e11;    // Young's modulus of pipe material (Pa) - steel default
    Real D = pipe->diameter();
    Real e = 0.01;    // Wall thickness (m) - would need to add this property
    
    Real a = std::sqrt(K / (rho * (1 + (K * D) / (E * e))));
    
    return a;
}

void TransientSolver::apply_events(Real current_time) {
    for (const auto& event : events_) {
        if (current_time >= event.start_time && 
            current_time <= event.start_time + event.duration) {
            
            switch (event.type) {
                case TransientEventType::VALVE_CLOSURE:
                case TransientEventType::VALVE_OPENING:
                    apply_valve_event(event, current_time);
                    break;
                case TransientEventType::PUMP_START:
                case TransientEventType::PUMP_STOP:
                    apply_pump_event(event, current_time);
                    break;
                case TransientEventType::DEMAND_CHANGE:
                    apply_demand_event(event, current_time);
                    break;
            }
        }
    }
}

void TransientSolver::apply_valve_event(const TransientEvent& event, Real current_time) {
    // Find the valve/pipe
    auto pipes = network_->pipes();
    if (pipes.count(event.component_id) == 0) return;
    
    auto pipe = pipes.at(event.component_id);
    Real valve_opening = event.value_at_time(current_time);
    
    // Modify pipe resistance based on valve opening
    // This is simplified - actual implementation would modify the system equations
    Real base_resistance = 1.0;  // Would calculate from pipe properties
    Real valve_resistance = base_resistance / (valve_opening * valve_opening + 0.001);
    
    // Apply to flow calculation
    // In practice, this would modify the system matrix
}

void TransientSolver::store_solution_snapshot(Real time) {
    if (!transient_config_.save_time_history) return;
    
    history_.time_points.push_back(time);
    
    // Store node pressures
    size_t idx = 0;
    for (const auto& [id, node] : network_->nodes()) {
        history_.node_pressure_history[id].push_back(current_pressures_(idx));
        idx++;
    }
    
    // Store pipe flows
    idx = 0;
    for (const auto& [id, pipe] : network_->pipes()) {
        history_.pipe_flow_history[id].push_back(current_flows_(idx));
        history_.pipe_velocity_history[id].push_back(current_flows_(idx) / pipe->area());
        idx++;
    }
}

void TransientSolver::initialize_solution_storage() {
    history_.time_points.clear();
    history_.node_pressure_history.clear();
    history_.pipe_flow_history.clear();
    history_.pipe_velocity_history.clear();
    
    // Pre-allocate vectors
    size_t estimated_steps = static_cast<size_t>(
        transient_config_.total_time / transient_config_.output_interval) + 2;
    
    for (const auto& [id, node] : network_->nodes()) {
        history_.node_pressure_history[id].reserve(estimated_steps);
    }
    
    for (const auto& [id, pipe] : network_->pipes()) {
        history_.pipe_flow_history[id].reserve(estimated_steps);
        history_.pipe_velocity_history[id].reserve(estimated_steps);
    }
}

void TransientSolver::analyze_water_hammer() {
    // Placeholder for water hammer analysis
    // Would calculate max pressure surges, compare with Joukowsky equation, etc.
}

bool TransientSolver::check_courant_condition(Real dt) {
    for (const auto& [id, pipe] : network_->pipes()) {
        Real wave_speed = transient_config_.wave_speed;
        Real velocity = std::abs(pipe->velocity());
        Real dx = pipe->length();
        
        Real courant = (wave_speed + velocity) * dt / dx;
        if (courant > 1.0) return false;
    }
    return true;
}

bool TransientSolver::check_diffusion_condition(Real dt) {
    // Check diffusion stability if needed
    return true;
}

void TransientSolver::reset() {
    Solver::reset();
    events_.clear();
    history_.time_points.clear();
    history_.node_pressure_history.clear();
    history_.pipe_flow_history.clear();
    history_.pipe_velocity_history.clear();
}

// Implement remaining stub methods
void TransientSolver::explicit_euler_step(Real dt) {
    // Placeholder - would implement explicit Euler time stepping
}

void TransientSolver::implicit_euler_step(Real dt) {
    // Placeholder - would implement implicit Euler time stepping
}

void TransientSolver::runge_kutta_4_step(Real dt) {
    // Placeholder - would implement RK4 time stepping
}

void TransientSolver::apply_pump_event(const TransientEvent& event, Real current_time) {
    // Placeholder - would modify pump operating point
}

void TransientSolver::apply_demand_event(const TransientEvent& event, Real current_time) {
    // Placeholder - would modify node demand
}

Real TransientSolver::calculate_wave_speed_wylie(const Ptr<Pipe>& pipe) {
    // Alternative wave speed formula
    return calculate_wave_speed_wood(pipe);
}

void TransientSolver::build_mass_matrix(SparseMatrix& M) {
    // Placeholder - would build mass matrix for transient analysis
}

Real TransientSolver::calculate_joukowsky_pressure(const Ptr<Pipe>& pipe, Real velocity_change) {
    Real wave_speed = transient_config_.wave_speed;
    if (transient_config_.calculate_wave_speed) {
        wave_speed = calculate_wave_speed_wood(pipe);
    }
    
    return fluid_.mixture_density() * wave_speed * velocity_change;
}

void TransientSolver::apply_transient_boundary_conditions(Real current_time) {
    // Placeholder - would apply time-dependent boundary conditions
}

void TransientSolver::apply_reservoir_boundary(const Ptr<Node>& node) {
    // Placeholder - constant pressure boundary
}

void TransientSolver::apply_valve_boundary(const Ptr<Node>& node, Real opening) {
    // Placeholder - valve boundary condition
}

} // namespace pipeline_sim
