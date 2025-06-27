#include "pipeline_sim/transient_solver.h"
#include <iostream>
#include <iomanip>
#include <fstream>

namespace pipeline_sim {

TransientSolver::TransientSolver(Ptr<Network> network, const FluidProperties& fluid)
    : Solver(network, fluid),
      time_step_(1.0),
      simulation_time_(100.0),
      current_time_(0.0),
      output_interval_(10.0),
      time_scheme_(TimeScheme::IMPLICIT_EULER),
      wave_speed_method_(WaveSpeedMethod::HOMOGENEOUS) {
    
    // Allocate solution vectors
    size_t n = network->nodes().size() + network->pipes().size();
    solution_old_ = Vector::Zero(n);
    solution_old2_ = Vector::Zero(n);
}

SolutionResults TransientSolver::solve() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    SolutionResults results;
    
    // Initialize with steady-state
    std::cout << "Computing initial steady-state solution..." << std::endl;
    SteadyStateSolver steady_solver(network_, fluid_);
    steady_solver.config() = config_;
    auto steady_results = steady_solver.solve();
    
    if (!steady_results.converged) {
        std::cerr << "Failed to compute initial steady-state!" << std::endl;
        return results;
    }
    
    current_time_ = 0.0;
    double next_output_time = 0.0;
    
    // Calculate wave speeds and check CFL
    calculate_wave_speeds();
    check_cfl_condition();
    
    // Initialize solution history
    history_.clear();
    
    // Store initial conditions
    size_t n = network_->nodes().size() + network_->pipes().size();
    Vector x(n);
    
    // Copy steady-state solution
    size_t idx = 0;
    for (const auto& [id, node] : network_->nodes()) {
        x(idx++) = node->pressure();
    }
    for (const auto& [id, pipe] : network_->pipes()) {
        x(idx++) = pipe->flow_rate();
    }
    
    solution_old_ = x;
    solution_old2_ = x;
    
    // Save initial state
    update_solution(solution_old_);
    save_to_history();
    
    // Open output file if specified
    if (!output_file_.empty()) {
        output_stream_.open(output_file_);
        write_output_header();
        write_output_state();
    }
    
    // Time stepping loop
    int step = 0;
    while (current_time_ < simulation_time_) {
        step++;
        
        // Process events
        process_events();
        
        // Build and solve system
        SparseMatrix A(n, n);
        Vector b(n);
        
        build_system_matrix(A, b);
        
        // Solve linear system
        Eigen::SparseLU<SparseMatrix> solver;
        solver.compute(A);
        
        if (solver.info() != Eigen::Success) {
            std::cerr << "Matrix decomposition failed at t=" << current_time_ << std::endl;
            break;
        }
        
        Vector x_new = solver.solve(b);
        
        if (solver.info() != Eigen::Success) {
            std::cerr << "Solution failed at t=" << current_time_ << std::endl;
            break;
        }
        
        // Update solution
        update_solution(x_new);
        
        // Check convergence (for implicit schemes)
        Vector residual = A * x_new - b;
        results.residual = residual.norm();
        
        if (config_.verbose && step % 10 == 0) {
            std::cout << "Time step " << step << ": t=" << current_time_ 
                     << ", residual=" << results.residual << std::endl;
        }
        
        // Save to history
        if (current_time_ >= next_output_time) {
            save_to_history();
            write_output_state();
            next_output_time += output_interval_;
        }
        
        // Prepare for next time step
        solution_old2_ = solution_old_;
        solution_old_ = x_new;
        current_time_ += time_step_;
    }
    
    // Close output file
    if (output_stream_.is_open()) {
        output_stream_.close();
    }
    
    // Prepare results
    results.converged = true;
    results.iterations = step;
    
    // Store final state
    for (const auto& [id, node] : network_->nodes()) {
        results.node_pressures[id] = node->pressure();
        results.node_temperatures[id] = node->temperature();
    }
    
    for (const auto& [id, pipe] : network_->pipes()) {
        results.pipe_flow_rates[id] = pipe->flow_rate();
        results.pipe_pressure_drops[id] = pipe->upstream()->pressure() - pipe->downstream()->pressure();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    results.computation_time = std::chrono::duration<Real>(end_time - start_time).count();
    
    return results;
}

void TransientSolver::build_system_matrix(SparseMatrix& A, Vector& b) {
    // Build the transient system matrix based on time scheme
    
    switch (time_scheme_) {
        case TimeScheme::IMPLICIT_EULER:
            // Use the parent class method for spatial discretization
            {
                SparseMatrix A_spatial(A.rows(), A.cols());
                Vector b_spatial(b.size());
                
                // Create temporary steady solver to access build_system_matrix
                SteadyStateSolver temp_solver(network_, fluid_);
                temp_solver.build_system_matrix(A_spatial, b_spatial);
                
                // Add time derivative terms
                // ... (implementation details)
            }
            break;
            
        case TimeScheme::CRANK_NICOLSON:
            // Crank-Nicolson implementation
            break;
            
        case TimeScheme::BDF2:
            // BDF2 implementation
            break;
    }
}

void TransientSolver::calculate_wave_speeds() {
    for (const auto& [id, pipe] : network_->pipes()) {
        Real a = 0.0;
        
        switch (wave_speed_method_) {
            case WaveSpeedMethod::HOMOGENEOUS:
                // Simplified homogeneous model
                a = 1000.0; // m/s (typical for water)
                break;
                
            case WaveSpeedMethod::WALLIS:
                // Wallis correlation
                break;
                
            case WaveSpeedMethod::MEASURED:
                // Use measured values
                break;
        }
        
        wave_speeds_[id] = a;
    }
}

void TransientSolver::check_cfl_condition() const {
    Real max_cfl = 0.0;
    
    for (const auto& [id, pipe] : network_->pipes()) {
        Real a = wave_speeds_.at(id);
        Real dx = pipe->length() / 10.0; // Assume 10 segments
        Real cfl = a * time_step_ / dx;
        
        if (cfl > max_cfl) {
            max_cfl = cfl;
        }
        
        if (cfl > 1.0 && config_.verbose) {
            std::cout << "Warning: CFL = " << cfl << " > 1.0 for pipe " << id << std::endl;
        }
    }
    
    if (config_.verbose) {
        std::cout << "Maximum CFL number: " << max_cfl << std::endl;
    }
}

void TransientSolver::process_events() {
    for (auto& event : events_) {
        if (std::abs(event.time - current_time_) < 0.5 * time_step_) {
            event.apply(network_, current_time_);
            
            if (config_.verbose) {
                std::cout << "Applied event at t=" << current_time_ << std::endl;
            }
        }
    }
}

void TransientSolver::save_to_history() {
    TimeStep step;
    step.time = current_time_;
    
    for (const auto& [id, node] : network_->nodes()) {
        step.node_data[id] = {node->pressure(), node->temperature()};
    }
    
    for (const auto& [id, pipe] : network_->pipes()) {
        step.pipe_data[id] = {pipe->flow_rate(), pipe->velocity()};
    }
    
    history_.push_back(step);
}

void TransientSolver::write_output_header() {
    if (!output_stream_.is_open()) return;
    
    output_stream_ << "Time";
    
    for (const auto& [id, node] : network_->nodes()) {
        output_stream_ << "," << id << "_pressure";
    }
    
    for (const auto& [id, pipe] : network_->pipes()) {
        output_stream_ << "," << id << "_flow";
    }
    
    output_stream_ << std::endl;
}

void TransientSolver::write_output_state() {
    if (!output_stream_.is_open()) return;
    
    output_stream_ << current_time_;
    
    for (const auto& [id, node] : network_->nodes()) {
        output_stream_ << "," << node->pressure();
    }
    
    for (const auto& [id, pipe] : network_->pipes()) {
        output_stream_ << "," << pipe->flow_rate();
    }
    
    output_stream_ << std::endl;
}

void TransientSolver::update_solution(const Vector& x) {
    size_t idx = 0;
    
    // Update node pressures
    for (const auto& [id, node] : network_->nodes()) {
        node->set_pressure(x(idx++));
    }
    
    // Update pipe flows
    for (const auto& [id, pipe] : network_->pipes()) {
        pipe->set_flow_rate(x(idx++));
    }
}

bool TransientSolver::check_convergence(const Vector& residual) {
    return residual.norm() < config_.tolerance;
}

} // namespace pipeline_sim
