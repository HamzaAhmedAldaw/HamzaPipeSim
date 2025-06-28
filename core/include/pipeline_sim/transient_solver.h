// ===== transient_solver.h =====
#ifndef PIPELINE_SIM_TRANSIENT_SOLVER_H
#define PIPELINE_SIM_TRANSIENT_SOLVER_H

#include "pipeline_sim/solver.h"
#include <vector>
#include <memory>

namespace pipeline_sim {

// Time integration schemes
enum class TimeIntegrationScheme {
    EXPLICIT_EULER,
    IMPLICIT_EULER,
    CRANK_NICOLSON,
    RUNGE_KUTTA_4
};

// Transient solver configuration
struct TransientSolverConfig : public SolverConfig {
    // Time stepping
    Real time_step = 0.1;  // seconds
    Real total_time = 100.0;  // seconds
    Real min_time_step = 1e-6;
    Real max_time_step = 1.0;
    bool adaptive_time_stepping = true;
    
    // Integration scheme
    TimeIntegrationScheme integration_scheme = TimeIntegrationScheme::CRANK_NICOLSON;
    
    // Stability parameters
    Real courant_number = 0.5;
    Real diffusion_number = 0.5;
    
    // Wave speed calculation
    Real wave_speed = 1000.0;  // m/s (default water hammer wave speed)
    bool calculate_wave_speed = true;
    
    // Output control
    Real output_interval = 1.0;  // seconds
    bool save_time_history = true;
};

// Time history data
struct TimeHistoryData {
    std::vector<Real> time_points;
    std::map<std::string, std::vector<Real>> node_pressure_history;
    std::map<std::string, std::vector<Real>> pipe_flow_history;
    std::map<std::string, std::vector<Real>> pipe_velocity_history;
};

// Transient event types
enum class TransientEventType {
    VALVE_CLOSURE,
    VALVE_OPENING,
    PUMP_START,
    PUMP_STOP,
    DEMAND_CHANGE,
    PRESSURE_SURGE
};

// Transient event
struct TransientEvent {
    TransientEventType type;
    std::string component_id;
    Real start_time;
    Real duration;
    Real initial_value;
    Real final_value;
    
    // Constructor
    TransientEvent(TransientEventType t, const std::string& id, 
                  Real start, Real dur, Real init, Real final)
        : type(t), component_id(id), start_time(start), 
          duration(dur), initial_value(init), final_value(final) {}
          
    // Get value at time t
    Real value_at_time(Real t) const {
        if (t < start_time) return initial_value;
        if (t > start_time + duration) return final_value;
        
        Real fraction = (t - start_time) / duration;
        return initial_value + fraction * (final_value - initial_value);
    }
};

// Transient solution results
struct TransientResults : public SolutionResults {
    TimeHistoryData time_history;
    Real max_pressure_surge = 0.0;
    Real min_pressure = 1e10;
    std::string max_surge_location;
    Real max_surge_time = 0.0;
    
    // Water hammer analysis
    Real theoretical_joukowsky_pressure = 0.0;
    Real actual_max_water_hammer = 0.0;
    Real damping_factor = 0.0;
};

// Transient solver class
class TransientSolver : public Solver {
public:
    TransientSolver(Ptr<Network> network, const FluidProperties& fluid);
    
    // Main solver interface
    SolutionResults solve() override;
    void reset() override;
    
    // Transient-specific methods
    TransientResults solve_transient();
    
    // Event management
    void add_event(const TransientEvent& event) { events_.push_back(event); }
    void clear_events() { events_.clear(); }
    const std::vector<TransientEvent>& events() const { return events_; }
    
    // Configuration
    void set_transient_config(const TransientSolverConfig& config) { 
        transient_config_ = config; 
        // Also update base config
        config_ = config;
    }
    const TransientSolverConfig& transient_config() const { return transient_config_; }
    
protected:
    // Time stepping methods
    bool advance_time_step(Real dt, Real current_time);
    Real calculate_stable_time_step(Real current_time);
    
    // Integration schemes
    void explicit_euler_step(Real dt);
    void implicit_euler_step(Real dt);
    void crank_nicolson_step(Real dt);
    void runge_kutta_4_step(Real dt);
    
    // System assembly methods
    void build_mass_matrix(SparseMatrix& M);
    void build_transient_system(SparseMatrix& A, Vector& b, Real dt, Real theta);
    
    // Wave speed calculations
    Real calculate_wave_speed_wood(const Ptr<Pipe>& pipe);
    Real calculate_wave_speed_wylie(const Ptr<Pipe>& pipe);
    
    // Event handling
    void apply_events(Real current_time);
    void apply_valve_event(const TransientEvent& event, Real current_time);
    void apply_pump_event(const TransientEvent& event, Real current_time);
    void apply_demand_event(const TransientEvent& event, Real current_time);
    
    // Solution storage
    void store_solution_snapshot(Real time);
    void initialize_solution_storage();
    
    // Analysis methods
    void analyze_water_hammer();
    Real calculate_joukowsky_pressure(const Ptr<Pipe>& pipe, Real velocity_change);
    
    // Boundary conditions
    void apply_transient_boundary_conditions(Real current_time);
    void apply_reservoir_boundary(const Ptr<Node>& node);
    void apply_valve_boundary(const Ptr<Node>& node, Real opening);
    
    // Stability checks
    bool check_courant_condition(Real dt);
    bool check_diffusion_condition(Real dt);
    
protected:
    TransientSolverConfig transient_config_;
    std::vector<TransientEvent> events_;
    
    // Solution storage
    Vector current_pressures_;
    Vector current_flows_;
    Vector previous_pressures_;
    Vector previous_flows_;
    
    // Time history storage
    TimeHistoryData history_;
    
    // Steady-state solver for initialization
    std::unique_ptr<SteadyStateSolver> steady_solver_;
};

} // namespace pipeline_sim

#endif // PIPELINE_SIM_TRANSIENT_SOLVER_H