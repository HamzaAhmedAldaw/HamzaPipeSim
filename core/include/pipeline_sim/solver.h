#pragma once

#include "pipeline_sim/types.h"
#include "pipeline_sim/network.h"
#include "pipeline_sim/fluid_properties.h"
#include <chrono>
#include <map>
#include <vector>

namespace pipeline_sim {

/// Solver configuration
struct SolverConfig {
    Real tolerance{1e-6};
    int max_iterations{100};
    Real relaxation_factor{1.0};
    bool verbose{false};
};

/// Solution results
struct SolutionResults {
    bool converged{false};
    int iterations{0};
    Real residual{0.0};
    Real computation_time{0.0};
    
    std::map<std::string, Real> node_pressures;
    std::map<std::string, Real> node_temperatures;
    std::map<std::string, Real> pipe_flow_rates;
    std::map<std::string, Real> pipe_pressure_drops;
    
    /// Get pressure drop for a pipe
    Real pressure_drop(const Ptr<Pipe>& pipe) const;
    
    /// Get outlet pressure for a pipe
    Real outlet_pressure(const Ptr<Pipe>& pipe) const;
};

/// Base solver class
class Solver {
public:
    Solver(Ptr<Network> network, const FluidProperties& fluid);
    virtual ~Solver() = default;
    
    /// Solve the network
    virtual SolutionResults solve();
    
    /// Get/set configuration
    SolverConfig& config() { return config_; }
    const SolverConfig& config() const { return config_; }
    
protected:
    Ptr<Network> network_;
    FluidProperties fluid_;
    SolverConfig config_;
    
    /// Build system matrix (pure virtual)
    virtual void build_system_matrix(SparseMatrix& A, Vector& b) = 0;
    
    /// Update solution from solution vector
    virtual void update_solution(const Vector& x) = 0;
    
    /// Check convergence
    virtual bool check_convergence(const Vector& residual) = 0;
};

/// Steady-state solver
class SteadyStateSolver : public Solver {
public:
    using Solver::Solver;
    
    SolutionResults solve() override;
    
protected:
    void build_system_matrix(SparseMatrix& A, Vector& b) override;
    void apply_boundary_conditions(SparseMatrix& A, Vector& b);
    void update_solution(const Vector& x) override;
    bool check_convergence(const Vector& residual) override;
    
    Real calculate_pressure_drop(const Ptr<Pipe>& pipe);
};

// Forward declaration - TransientSolver is defined in transient_solver.h
class TransientSolver;

} // namespace pipeline_sim