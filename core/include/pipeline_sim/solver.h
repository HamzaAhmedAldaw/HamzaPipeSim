#pragma once

#include "pipeline_sim/types.h"
#include "pipeline_sim/network.h"
#include "pipeline_sim/fluid_properties.h"
#include <chrono>
#include <map>
#include <vector>
#include <memory>

namespace pipeline_sim {

/// Advanced solver configuration for commercial use
struct SolverConfig {
    // Convergence parameters
    Real tolerance{1e-8};                    // Convergence tolerance
    Real flow_tolerance{1e-10};              // Flow balance tolerance
    int max_iterations{200};                 // Maximum iterations
    Real relaxation_factor{1.0};             // General relaxation
    
    // Advanced parameters
    Real min_flow_velocity{1e-8};            // Minimum velocity for numerical stability
    Real pressure_damping{0.8};              // Pressure update damping
    Real flow_damping{0.7};                  // Flow update damping for mixed BC
    
    // Laminar flow handling
    bool enable_laminar_correction{true};    // Special handling for low Re
    Real laminar_transition_Re{2300.0};      // Reynolds number for transition
    Real critical_zone_factor{1.2};          // Smooth transition zone multiplier
    
    // Solver options
    bool verbose{false};                     // Detailed output
    bool adaptive_damping{true};             // Adjust damping based on convergence
    bool use_previous_solution{true};        // Warm start from previous solve
    
    // Numerical parameters
    Real min_pressure{1000.0};               // Minimum allowable pressure (Pa)
    Real max_pressure{1e8};                  // Maximum allowable pressure (Pa)
    Real jacobian_epsilon{1e-8};             // Perturbation for numerical derivatives
};

/// Detailed solution results
struct SolutionResults {
    // Convergence information
    bool converged{false};
    int iterations{0};
    Real residual{0.0};
    Real max_pressure_change{0.0};
    Real max_flow_change{0.0};
    Real computation_time{0.0};
    
    // Solution maps
    std::map<std::string, Real> node_pressures;
    std::map<std::string, Real> node_temperatures;
    std::map<std::string, Real> pipe_flow_rates;
    std::map<std::string, Real> pipe_pressure_drops;
    std::map<std::string, Real> pipe_velocities;
    std::map<std::string, Real> pipe_reynolds_numbers;
    std::map<std::string, Real> pipe_friction_factors;
    
    // Mass balance information
    std::map<std::string, Real> node_mass_imbalance;
    Real max_mass_imbalance{0.0};
    
    // Methods
    Real pressure_drop(const Ptr<Pipe>& pipe) const;
    Real outlet_pressure(const Ptr<Pipe>& pipe) const;
    Real mass_balance_error() const { return max_mass_imbalance; }
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
    
    /// Reset solver state
    virtual void reset();
    
protected:
    Ptr<Network> network_;
    FluidProperties fluid_;
    SolverConfig config_;
    
    // Previous solution for warm start
    std::map<std::string, Real> prev_pressures_;
    std::map<std::string, Real> prev_flows_;
    
    /// Build system matrix (pure virtual)
    virtual void build_system_matrix(SparseMatrix& A, Vector& b) = 0;
    
    /// Update solution from solution vector
    virtual void update_solution(const Vector& x) = 0;
    
    /// Check convergence
    virtual bool check_convergence(const Vector& residual) = 0;
};

/// Commercial-grade steady-state solver
class SteadyStateSolver : public Solver {
public:
    using Solver::Solver;
    
    SolutionResults solve() override;
    void reset() override;
    
protected:
    // Legacy interface (not used in new implementation)
    void build_system_matrix(SparseMatrix& A, Vector& b) override;
    void apply_boundary_conditions(SparseMatrix& A, Vector& b);
    void update_solution(const Vector& x) override;
    bool check_convergence(const Vector& residual) override;
    Real calculate_pressure_drop(const Ptr<Pipe>& pipe);
    
private:
    /// Main solver using unified formulation
    bool solve_unified_formulation(int& iterations_performed, Real& final_residual);
    
    /// Build Jacobian for unified pressure-flow formulation
    bool build_unified_jacobian(SparseMatrix& J, Vector& F,
                               const Vector& x,
                               const std::vector<std::string>& var_names,
                               const std::map<std::string, size_t>& var_indices);
    
    /// Calculate pipe flow with advanced friction model
    void calculate_pipe_flow_advanced(const Ptr<Pipe>& pipe,
                                     Real p_upstream, Real p_downstream,
                                     Real& Q, Real& dQ_dp_up, Real& dQ_dp_down,
                                     Real& friction_factor);
    
    /// Laminar flow correction for low Reynolds numbers
    Real laminar_correction_factor(Real Re, Real& dFactor_dRe);
    
    /// Advanced friction factor calculation
    Real calculate_friction_factor_advanced(Real Re, Real relative_roughness,
                                          Real& df_dRe);
    
    /// Colebrook-White equation solver
    Real solve_colebrook_white(Real Re, Real relative_roughness,
                              Real initial_guess = 0.02);
    
    /// Update network state from solution vector
    void update_network_from_solution(const Vector& x,
                                     const std::vector<std::string>& var_names,
                                     const std::map<std::string, size_t>& var_indices);
    
    /// Extract comprehensive results
    void extract_detailed_results(SolutionResults& results);
    
    /// Check and enforce solution bounds
    void enforce_solution_bounds(Vector& x,
                                const std::vector<std::string>& var_names);
    
    /// Adaptive damping based on convergence behavior
    Real calculate_adaptive_damping(Real base_damping, int iteration,
                                   Real current_residual, Real prev_residual);
    
    /// Initialize solution with smart guessing
    void initialize_solution_advanced(Vector& x,
                                     const std::vector<std::string>& var_names,
                                     const std::map<std::string, size_t>& var_indices);
};

// Forward declaration
class TransientSolver;

} // namespace pipeline_sim