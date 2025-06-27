#include "pipeline_sim/transient_solver.h"
#include <iostream>
#include <iomanip>
#include <fstream>

namespace pipeline_sim {

// Implementation of event classes
void ValveClosureEvent::apply(Network& network, Real time) {
    // Find valve and update opening
    Real progress = (time - start_time_) / duration_;
    progress = std::max(0.0, std::min(1.0, progress));
    
    Real current_opening = 1.0 - progress * (1.0 - final_opening_);
    
    // Update valve opening in network
    // Note: This assumes valve is stored as equipment in the network
    for (const auto& [id, pipe] : network.pipes()) {
        if (pipe->has_valve() && pipe->valve_id() == valve_id_) {
            pipe->set_valve_opening(current_opening);
            break;
        }
    }
}

void PumpTripEvent::apply(Network& network, Real time) {
    if (!triggered_) {
        // Find pump and set speed to 0
        for (const auto& [id, node] : network.nodes()) {
            if (node->type() == NodeType::PUMP && node->id() == pump_id_) {
                node->set_pump_speed(0.0);
                triggered_ = true;
                break;
            }
        }
    }
}

// TransientSolver implementation
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
    bool cfl_ok = check_cfl_condition();
    if (!cfl_ok && config_.verbose) {
        std::cout << "Warning: CFL condition violated!" << std::endl;
    }
    
    // Initialize solution history
    history_.times.clear();
    history_.node_pressures.clear();
    history_.pipe_flows.clear();
    
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
    size_t n = A.rows();
    
    // First, get the spatial discretization from parent class
    SparseMatrix A_spatial(n, n);
    Vector b_spatial(n);
    
    // We need to access the steady-state matrix building
    // Since it's protected, we create a temporary public method or use a different approach
    // For now, we'll implement the matrix building directly here
    
    std::vector<Eigen::Triplet<Real>> triplets;
    
    // Build spatial discretization (simplified version)
    size_t node_idx = 0;
    size_t pipe_idx = network_->nodes().size();
    
    // Node equations (mass conservation)
    for (const auto& [node_id, node] : network_->nodes()) {
        // Mass conservation at node
        for (const auto& pipe : network_->get_upstream_pipes(node)) {
            size_t pidx = pipe_idx + std::distance(
                network_->pipes().begin(),
                network_->pipes().find(pipe->id())
            );
            triplets.push_back(Eigen::Triplet<Real>(
                static_cast<int>(node_idx), 
                static_cast<int>(pidx), 
                1.0));
        }
        
        for (const auto& pipe : network_->get_downstream_pipes(node)) {
            size_t pidx = pipe_idx + std::distance(
                network_->pipes().begin(),
                network_->pipes().find(pipe->id())
            );
            triplets.push_back(Eigen::Triplet<Real>(
                static_cast<int>(node_idx), 
                static_cast<int>(pidx), 
                -1.0));
        }
        
        node_idx++;
    }
    
    // Pipe momentum equations
    pipe_idx = network_->nodes().size();
    for (const auto& [pipe_id, pipe] : network_->pipes()) {
        size_t up_idx = std::distance(
            network_->nodes().begin(),
            network_->nodes().find(pipe->upstream()->id())
        );
        size_t down_idx = std::distance(
            network_->nodes().begin(),
            network_->nodes().find(pipe->downstream()->id())
        );
        
        // Momentum equation coefficients
        Real area = pipe->area();
        Real length = pipe->length();
        
        triplets.push_back(Eigen::Triplet<Real>(
            static_cast<int>(pipe_idx), 
            static_cast<int>(up_idx), 
            area));
        triplets.push_back(Eigen::Triplet<Real>(
            static_cast<int>(pipe_idx), 
            static_cast<int>(down_idx), 
            -area));
        
        pipe_idx++;
    }
    
    A_spatial.setFromTriplets(triplets.begin(), triplets.end());
    
    // Now apply time discretization
    switch (time_scheme_) {
        case TimeScheme::IMPLICIT_EULER:
            // (I - dt*A) * x_new = x_old
            A = SparseMatrix(n, n);
            A.setIdentity();
            A -= time_step_ * A_spatial;
            b = solution_old_;
            break;
            
        case TimeScheme::CRANK_NICOLSON:
            // (I - 0.5*dt*A) * x_new = (I + 0.5*dt*A) * x_old
            A = SparseMatrix(n, n);
            A.setIdentity();
            A -= 0.5 * time_step_ * A_spatial;
            
            b = solution_old_ + 0.5 * time_step_ * A_spatial * solution_old_;
            break;
            
        case TimeScheme::RUNGE_KUTTA_4:
            // RK4 is explicit, so we use identity matrix
            A = SparseMatrix(n, n);
            A.setIdentity();
            // RK4 implementation would go here
            b = solution_old_;
            break;
            
        default:
            // Default to implicit Euler
            A = SparseMatrix(n, n);
            A.setIdentity();
            A -= time_step_ * A_spatial;
            b = solution_old_;
            break;
    }
}

void TransientSolver::calculate_wave_speeds() {
    for (const auto& [id, pipe] : network_->pipes()) {
        Real a = 0.0;
        
        // Use string comparison for wave speed method
        if (wave_speed_method_ == "homogeneous" || wave_speed_method_ == "automatic") {
            // Simplified homogeneous model
            Real bulk_modulus = 2.2e9; // Pa (water)
            Real density = fluid_.mixture_density();
            Real pipe_modulus = 200e9; // Pa (steel)
            Real wall_thickness = 0.01; // m (assumed)
            Real diameter = pipe->diameter();
            
            // Korteweg formula
            Real K_eff = 1.0 / (1.0/bulk_modulus + diameter/(wall_thickness * pipe_modulus));
            a = std::sqrt(K_eff / density);
        }
        else if (wave_speed_method_ == "wallis") {
            // Wallis correlation for two-phase flow
            Real gas_frac = fluid_.gas_fraction;
            Real liquid_density = fluid_.oil_density * fluid_.oil_fraction + 
                                fluid_.water_density * fluid_.water_fraction;
            Real gas_density = fluid_.gas_density;
            Real pressure = 1e6; // Pa (assumed)
            
            Real a_liquid = 1000.0; // m/s
            Real a_gas = std::sqrt(1.4 * pressure / gas_density);
            
            // Wallis model
            a = 1.0 / std::sqrt(
                gas_frac * (1 - gas_frac) * 
                (1.0/(gas_density * a_gas * a_gas) + 1.0/(liquid_density * a_liquid * a_liquid)) +
                gas_frac / (gas_density * a_gas * a_gas) +
                (1 - gas_frac) / (liquid_density * a_liquid * a_liquid)
            );
        }
        else if (wave_speed_method_ == "measured") {
            // Use default value or lookup from data
            a = 1000.0; // m/s
        }
        else {
            // Default
            a = 1000.0; // m/s
        }
        
        wave_speeds_[id] = a;
    }
}

bool TransientSolver::check_cfl_condition() const {
    Real max_cfl = 0.0;
    std::string critical_pipe;
    
    for (const auto& [id, pipe] : network_->pipes()) {
        auto wave_speed_it = wave_speeds_.find(id);
        if (wave_speed_it == wave_speeds_.end()) continue;
        
        Real a = wave_speed_it->second;
        Real dx = pipe->length() / 10.0; // Assume 10 segments
        Real cfl = a * time_step_ / dx;
        
        if (cfl > max_cfl) {
            max_cfl = cfl;
            critical_pipe = id;
        }
        
        if (cfl > 1.0 && config_.verbose) {
            std::cout << "Warning: CFL = " << cfl << " > 1.0 for pipe " << id << std::endl;
        }
    }
    
    if (config_.verbose) {
        std::cout << "Maximum CFL number: " << max_cfl << " in pipe " << critical_pipe << std::endl;
    }
    
    return max_cfl <= 1.0;
}

void TransientSolver::process_events() {
    for (auto& event : events_) {
        if (event->should_trigger(current_time_)) {
            event->apply(*network_, current_time_);
            
            if (config_.verbose) {
                std::cout << "Applied event: " << event->description() 
                         << " at t=" << current_time_ << std::endl;
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

void TransientSolver::apply_method_of_characteristics(SparseMatrix& A, Vector& b) {
    // Method of characteristics implementation
    // This is a placeholder - full MOC implementation would be quite complex
    // For now, we use the finite difference approach in build_system_matrix
}

// LinePackCalculator implementation
LinePackCalculator::LinePackResult LinePackCalculator::calculate(
    const Network& network,
    const SolutionResults& results,
    const FluidProperties& fluid
) {
    LinePackResult result;
    result.total_mass = 0.0;
    result.total_volume = 0.0;
    
    Real total_pressure = 0.0;
    int pressure_count = 0;
    
    for (const auto& [id, pipe] : network.pipes()) {
        Real volume = pipe->area() * pipe->length();
        Real avg_pressure = (pipe->upstream()->pressure() + pipe->downstream()->pressure()) / 2.0;
        Real density = fluid.mixture_density(); // Could use pressure-dependent density
        Real mass = density * volume;
        
        result.pipe_masses[id] = mass;
        result.total_mass += mass;
        result.total_volume += volume;
        
        total_pressure += avg_pressure;
        pressure_count++;
    }
    
    result.average_pressure = pressure_count > 0 ? total_pressure / pressure_count : 0.0;
    result.average_density = result.total_volume > 0 ? result.total_mass / result.total_volume : 0.0;
    
    return result;
}

// SurgeAnalyzer implementation
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
    for (const auto& [node_id, pressure_history] : history.node_pressures) {
        for (Real pressure : pressure_history) {
            if (pressure > result.max_pressure) {
                result.max_pressure = pressure;
                result.max_location = node_id;
            }
            if (pressure < result.min_pressure) {
                result.min_pressure = pressure;
                result.min_location = node_id;
            }
        }
    }
    
    result.surge_pressure = result.max_pressure - result.min_pressure;
    result.exceeds_mawp = result.max_pressure > mawp;
    
    return result;
}

} // namespace pipeline_sim
