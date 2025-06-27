/// AI_GENERATED: Transient solver and ML integration
/// Generated on: 2025-06-27

// ===== include/pipeline_sim/transient_solver.h =====
#pragma once

#include "pipeline_sim/solver.h"
#include <deque>
#include <fstream>

namespace pipeline_sim {

/// Time integration schemes
enum class TimeScheme {
    EXPLICIT_EULER,
    IMPLICIT_EULER,
    CRANK_NICOLSON,
    RUNGE_KUTTA_4
};

/// Transient event types
class TransientEvent {
public:
    virtual ~TransientEvent() = default;
    
    /// Check if event should trigger
    virtual bool should_trigger(Real time) const = 0;
    
    /// Apply event to network
    virtual void apply(Network& network, Real time) = 0;
    
    /// Get event description
    virtual std::string description() const = 0;
};

/// Valve closure event
class ValveClosureEvent : public TransientEvent {
public:
    ValveClosureEvent(const std::string& valve_id, 
                     Real start_time, 
                     Real duration,
                     Real final_opening = 0.0)
        : valve_id_(valve_id), 
          start_time_(start_time),
          duration_(duration),
          final_opening_(final_opening) {}
    
    bool should_trigger(Real time) const override {
        return time >= start_time_ && time <= start_time_ + duration_;
    }
    
    void apply(Network& network, Real time) override;
    
    std::string description() const override {
        return "Valve closure: " + valve_id_;
    }
    
private:
    std::string valve_id_;
    Real start_time_;
    Real duration_;
    Real final_opening_;
};

/// Pump trip event
class PumpTripEvent : public TransientEvent {
public:
    PumpTripEvent(const std::string& pump_id, Real trip_time)
        : pump_id_(pump_id), trip_time_(trip_time) {}
    
    bool should_trigger(Real time) const override {
        return time >= trip_time_ && !triggered_;
    }
    
    void apply(Network& network, Real time) override;
    
    std::string description() const override {
        return "Pump trip: " + pump_id_;
    }
    
private:
    std::string pump_id_;
    Real trip_time_;
    mutable bool triggered_{false};
};

/// Transient solver with method of characteristics
class TransientSolver : public Solver {
public:
    using Solver::Solver;
    
    /// Set time integration parameters
    void set_time_step(Real dt) { time_step_ = dt; }
    void set_simulation_time(Real t) { simulation_time_ = t; }
    void set_time_scheme(TimeScheme scheme) { time_scheme_ = scheme; }
    
    /// Set wave speed calculation method
    void set_wave_speed_method(const std::string& method) {
        wave_speed_method_ = method;
    }
    
    /// Add transient event
    void add_event(std::unique_ptr<TransientEvent> event) {
        events_.push_back(std::move(event));
    }
    
    /// Set output parameters
    void set_output_interval(Real interval) { output_interval_ = interval; }
    void set_output_file(const std::string& filename) {
        output_file_ = filename;
    }
    
    /// Run transient simulation
    SolutionResults solve() override;
    
    /// Get time history data
    struct TimeHistory {
        std::vector<Real> times;
        std::map<std::string, std::vector<Real>> node_pressures;
        std::map<std::string, std::vector<Real>> pipe_flows;
    };
    
    const TimeHistory& get_time_history() const { return history_; }
    
protected:
    void build_system_matrix(SparseMatrix& A, Vector& b) override;
    void update_solution(const Vector& x) override;
    bool check_convergence(const Vector& residual) override;
    
private:
    Real time_step_{0.1};
    Real simulation_time_{3600.0};
    Real current_time_{0.0};
    TimeScheme time_scheme_{TimeScheme::IMPLICIT_EULER};
    std::string wave_speed_method_{"automatic"};
    
    Real output_interval_{1.0};
    std::string output_file_;
    std::ofstream output_stream_;
    
    std::vector<std::unique_ptr<TransientEvent>> events_;
    TimeHistory history_;
    
    // Previous time step solution
    Vector solution_old_;
    Vector solution_old2_;  // For higher-order schemes
    
    /// Calculate wave speed for each pipe
    void calculate_wave_speeds();
    std::map<std::string, Real> wave_speeds_;
    
    /// Apply method of characteristics
    void apply_method_of_characteristics(SparseMatrix& A, Vector& b);
    
    /// Check CFL condition
    bool check_cfl_condition() const;
    
    /// Process events
    void process_events();
    
    /// Save current state to history
    void save_to_history();
    
    /// Write output file header
    void write_output_header();
    
    /// Write current state to output
    void write_output_state();
};

/// Line pack calculation
class LinePackCalculator {
public:
    struct LinePackResult {
        Real total_mass;        // kg
        Real total_volume;      // m³
        Real average_pressure;  // Pa
        Real average_density;   // kg/m³
        std::map<std::string, Real> pipe_masses;
    };
    
    static LinePackResult calculate(
        const Network& network,
        const SolutionResults& results,
        const FluidProperties& fluid
    );
};

/// Surge analysis
class SurgeAnalyzer {
public:
    struct SurgeResult {
        Real max_pressure;
        Real min_pressure;
        std::string max_location;
        std::string min_location;
        Real surge_pressure;
        bool exceeds_mawp;
    };
    
    static SurgeResult analyze(
        const TransientSolver::TimeHistory& history,
        const Network& network,
        Real mawp  // Maximum allowable working pressure
    );
};

} // namespace pipeline_sim
