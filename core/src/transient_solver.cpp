
// ===== src/transient_solver.cpp =====
#include "pipeline_sim/transient_solver.h"
#include <iostream>
#include <iomanip>
#include <chrono>

namespace pipeline_sim {

void ValveClosureEvent::apply(Network& network, Real time) {
    Real elapsed = time - start_time_;
    Real progress = elapsed / duration_;
    
    // Linear valve closure
    Real current_opening = 1.0 - progress * (1.0 - final_opening_);
    
    // TODO: Apply to valve in network
    // auto valve = network.get_equipment(valve_id_);
    // valve->set_opening(current_opening);
}

void PumpTripEvent::apply(Network& network, Real time) {
    if (!triggered_) {
        triggered_ = true;
        // TODO: Stop pump
        // auto pump = network.get_equipment(pump_id_);
        // pump->set_speed_ratio(0.0);
    }
}

SolutionResults TransientSolver::solve() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    SolutionResults results;
    current_time_ = 0.0;
    
    // Initialize
    calculate_wave_speeds();
    if (!check_cfl_condition()) {
        std::cerr << "CFL condition violated! Reduce time step." << std::endl;
        return results;
    }
    
    // Get initial steady state
    SteadyStateSolver steady_solver(network_, fluid_);
    results = steady_solver.solve();
    
    if (!results.converged) {
        std::cerr << "Failed to obtain initial steady state!" << std::endl;
        return results;
    }
    
    // Initialize solution vectors
    size_t num_vars = network_->nodes().size() + network_->pipes().size();
    solution_old_.resize(num_vars);
    solution_old2_.resize(num_vars);
    
    // Store initial conditions
    update_solution(solution_old_);
    save_to_history();
    
    // Open output file
    if (!output_file_.empty()) {
        output_stream_.open(output_file_);
        write_output_header();
        write_output_state();
    }
    
    // Time stepping loop
    Real next_output = output_interval_;
    int step = 0;
    
    while (current_time_ < simulation_time_) {
        // Process events
        process_events();
        
        // Build and solve system
        SparseMatrix A(num_vars, num_vars);
        Vector b(num_vars);
        Vector x(num_vars);
        
        build_system_matrix(A, b);
        
        // Solve linear system
        Eigen::SparseLU<SparseMatrix> solver;
        solver.compute(A);
        
        if (solver.info() != Eigen::Success) {
            std::cerr << "Matrix decomposition failed at t=" 
                     << current_time_ << std::endl;
            break;
        }
        
        x = solver.solve(b);
        
        // Update solution
        update_solution(x);
        
        // Check for numerical instability
        Real max_pressure = 0.0;
        for (const auto& [id, pressure] : results.node_pressures) {
            max_pressure = std::max(max_pressure, std::abs(pressure));
        }
        
        if (max_pressure > 1e8) {  // 1000 bar
            std::cerr << "Numerical instability detected!" << std::endl;
            break;
        }
        
        // Output
        if (current_time_ >= next_output) {
            save_to_history();
            write_output_state();
            next_output += output_interval_;
            
            if (config_.verbose) {
                std::cout << "Time: " << std::setw(8) << std::fixed 
                         << std::setprecision(2) << current_time_ 
                         << " s" << std::endl;
            }
        }
        
        // Advance time
        solution_old2_ = solution_old_;
        solution_old_ = x;
        current_time_ += time_step_;
        step++;
    }
    
    // Close output file
    if (output_stream_.is_open()) {
        output_stream_.close();
    }
    
    // Final results
    results.converged = true;
    results.iterations = step;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    results.computation_time = 
        std::chrono::duration<Real>(end_time - start_time).count();
    
    return results;
}

void TransientSolver::build_system_matrix(SparseMatrix& A, Vector& b) {
    std::vector<Eigen::Triplet<Real>> triplets;
    
    const auto& nodes = network_->nodes();
    const auto& pipes = network_->pipes();
    
    // Apply method of characteristics
    if (wave_speed_method_ != "none") {
        apply_method_of_characteristics(A, b);
    }
    
    // Time discretization
    switch (time_scheme_) {
        case TimeScheme::IMPLICIT_EULER:
            // (I - dt*J)*x^{n+1} = x^n + dt*f
            // TODO: Implement implicit Euler
            break;
            
        case TimeScheme::CRANK_NICOLSON:
            // (I - 0.5*dt*J)*x^{n+1} = (I + 0.5*dt*J)*x^n + dt*f
            // TODO: Implement Crank-Nicolson
            break;
            
        default:
            // Fall back to steady-state formulation
            SteadyStateSolver::build_system_matrix(A, b);
    }
}

void TransientSolver::calculate_wave_speeds() {
    for (const auto& [id, pipe] : network_->pipes()) {
        Real K = 2.1e9;  // Bulk modulus of fluid (Pa)
        Real E = 200e9;  // Young's modulus of pipe (Pa)
        Real D = pipe->diameter();
        Real e = 0.01;   // Wall thickness (m)
        Real rho = fluid_.mixture_density();
        
        // Korteweg formula
        Real a = std::sqrt(K / (rho * (1 + K * D / (E * e))));
        
        wave_speeds_[id] = a;
    }
}

bool TransientSolver::check_cfl_condition() const {
    for (const auto& [id, pipe] : network_->pipes()) {
        Real a = wave_speeds_.at(id);
        Real dx = pipe->length() / 10;  // Assume 10 segments
        Real cfl = a * time_step_ / dx;
        
        if (cfl > 1.0) {
            std::cerr << "CFL violation in pipe " << id 
                     << ": CFL = " << cfl << std::endl;
            return false;
        }
    }
    return true;
}

void TransientSolver::process_events() {
    for (auto& event : events_) {
        if (event->should_trigger(current_time_)) {
            event->apply(*network_, current_time_);
            
            if (config_.verbose) {
                std::cout << "Event at t=" << current_time_ 
                         << ": " << event->description() << std::endl;
            }
        }
    }
}

void TransientSolver::save_to_history() {
    history_.times.push_back(current_time_);
    
    for (const auto& [id, node] : network_->nodes()) {
        history_.node_pressures[id].push_back(node->pressure());
    }
    
    for (const auto& [id, pipe] : network_->pipes()) {
        history_.pipe_flows[id].push_back(pipe->flow_rate());
    }
}

void TransientSolver::write_output_header() {
    output_stream_ << "Time";
    
    for (const auto& [id, node] : network_->nodes()) {
        output_stream_ << ",P_" << id;
    }
    
    for (const auto& [id, pipe] : network_->pipes()) {
        output_stream_ << ",Q_" << id;
    }
    
    output_stream_ << std::endl;
}

void TransientSolver::write_output_state() {
    output_stream_ << current_time_;
    
    for (const auto& [id, node] : network_->nodes()) {
        output_stream_ << "," << node->pressure();
    }
    
    for (const auto& [id, pipe] : network_->pipes()) {
        output_stream_ << "," << pipe->flow_rate();
    }
    
    output_stream_ << std::endl;
}

// Line Pack Calculator
LinePackCalculator::LinePackResult LinePackCalculator::calculate(
    const Network& network,
    const SolutionResults& results,
    const FluidProperties& fluid
) {
    LinePackResult result;
    result.total_mass = 0.0;
    result.total_volume = 0.0;
    
    Real total_pressure = 0.0;
    int num_pipes = 0;
    
    for (const auto& [id, pipe] : network.pipes()) {
        // Average pressure in pipe
        Real p_in = results.node_pressures.at(pipe->upstream()->id());
        Real p_out = results.node_pressures.at(pipe->downstream()->id());
        Real p_avg = (p_in + p_out) / 2.0;
        
        // Calculate density at average pressure
        Real density = fluid.mixture_density() * p_avg / constants::STANDARD_PRESSURE;
        
        // Line pack mass
        Real volume = pipe->volume();
        Real mass = density * volume;
        
        result.pipe_masses[id] = mass;
        result.total_mass += mass;
        result.total_volume += volume;
        total_pressure += p_avg;
        num_pipes++;
    }
    
    result.average_pressure = total_pressure / num_pipes;
    result.average_density = result.total_mass / result.total_volume;
    
    return result;
}

// Surge Analyzer
SurgeAnalyzer::SurgeResult SurgeAnalyzer::analyze(
    const TransientSolver::TimeHistory& history,
    const Network& network,
    Real mawp
) {
    SurgeResult result;
    result.max_pressure = 0.0;
    result.min_pressure = 1e10;
    result.exceeds_mawp = false;
    
    // Find maximum and minimum pressures
    for (const auto& [node_id, pressures] : history.node_pressures) {
        for (Real p : pressures) {
            if (p > result.max_pressure) {
                result.max_pressure = p;
                result.max_location = node_id;
            }
            if (p < result.min_pressure) {
                result.min_pressure = p;
                result.min_location = node_id;
            }
            if (p > mawp) {
                result.exceeds_mawp = true;
            }
        }
    }
    
    // Calculate surge pressure (Joukowsky equation)
    // This is simplified - actual calculation would be more complex
    result.surge_pressure = result.max_pressure - result.min_pressure;
    
    return result;
}

} // namespace pipeline_sim