/// AI_GENERATED: Solver interface and implementations
/// Generated on: 2025-06-27
#pragma once

#include "pipeline_sim/types.h"
#include "pipeline_sim/network.h"
#include "pipeline_sim/fluid_properties.h"
#include <memory>
#include <map>

namespace pipeline_sim {

/// Solver configuration
struct SolverConfig {
    Real tolerance{1e-6};
    int max_iterations{1000};
    Real relaxation_factor{0.7};
    bool verbose{false};
    bool use_parallel{true};
    int num_threads{0};  // 0 = auto-detect
};

/// Solution results
struct SolutionResults {
    bool converged{false};
    int iterations{0};
    Real residual{0.0};
    Real computation_time{0.0};
    
    // Node results
    std::map<std::string, Real> node_pressures;
    std::map<std::string, Real> node_temperatures;
    
    // Pipe results
    std::map<std::string, Real> pipe_flow_rates;
    std::map<std::string, Real> pipe_pressure_drops;
    std::map<std::string, Real> pipe_liquid_holdups;
    
    // Access helpers
    Real pressure_drop(const Ptr<Pipe>& pipe) const;
    Real outlet_pressure(const Ptr<Pipe>& pipe) const;
};

/// Base solver interface
class Solver {
public:
    Solver(Ptr<Network> network, const FluidProperties& fluid);
    virtual ~Solver() = default;
    
    /// Solve the network
    virtual SolutionResults solve() = 0;
    
    /// Get/set configuration
    SolverConfig& config() { return config_; }
    const SolverConfig& config() const { return config_; }
    
protected:
    Ptr<Network> network_;
    FluidProperties fluid_;
    SolverConfig config_;
    
    /// Build system matrix
    virtual void build_system_matrix(SparseMatrix& A, Vector& b) = 0;
    
    /// Update solution
    virtual void update_solution(const Vector& x) = 0;
    
    /// Check convergence
    virtual bool check_convergence(const Vector& residual) = 0;
};

/// Steady-state solver
class SteadyStateSolver : public Solver {
public:
    using Solver::Solver;
    
    SolutionResults solve() override;
    
private:
    void build_system_matrix(SparseMatrix& A, Vector& b) override;
    void update_solution(const Vector& x) override;
    bool check_convergence(const Vector& residual) override;
    
    /// Calculate pressure drop for a pipe
    Real calculate_pressure_drop(const Ptr<Pipe>& pipe);
    
    /// Apply boundary conditions
    void apply_boundary_conditions(SparseMatrix& A, Vector& b);
};

/// Transient solver (future implementation)
class TransientSolver : public Solver {
public:
    using Solver::Solver;
    
    void set_time_step(Real dt) { time_step_ = dt; }
    void set_simulation_time(Real t) { simulation_time_ = t; }
    
    SolutionResults solve() override;
    
private:
    Real time_step_{1.0};
    Real simulation_time_{3600.0};
    
    void build_system_matrix(SparseMatrix& A, Vector& b) override;
    void update_solution(const Vector& x) override;
    bool check_convergence(const Vector& residual) override;
};

} // namespace pipeline_sim