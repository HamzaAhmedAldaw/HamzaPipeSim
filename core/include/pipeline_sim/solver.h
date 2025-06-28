// ===== solver.h =====
#ifndef PIPELINE_SIM_SOLVER_H
#define PIPELINE_SIM_SOLVER_H

#include "pipeline_sim/types.h"
#include "pipeline_sim/network.h"
#include "pipeline_sim/fluid_properties.h"
#include <Eigen/Sparse>

namespace pipeline_sim {

// Forward declarations
class Pipe;
class Node;

// Solver configuration
struct SolverConfig {
    // Convergence criteria
    Real tolerance = 1e-6;
    Real relaxation_factor = 0.8;
    int max_iterations = 100;
    bool verbose = false;
    
    // Advanced parameters
    Real pressure_damping = 0.7;
    Real flow_damping = 0.8;
    bool adaptive_damping = true;
    bool use_previous_solution = true;
    
    // Numerical parameters
    Real min_pressure = 1e4;  // 0.1 bar
    Real max_pressure = 1e8;  // 1000 bar
    Real min_flow_velocity = 1e-6;  // m/s
    Real jacobian_epsilon = 1e-8;
    
    // Flow regime parameters
    Real laminar_transition_Re = 2300.0;
    Real critical_zone_factor = 1.5;
    bool enable_laminar_correction = true;
};

// Solution results
struct SolutionResults {
    bool converged = false;
    int iterations = 0;
    Real residual = 0.0;
    Real computation_time = 0.0;
    
    // Detailed results
    std::map<std::string, Real> node_pressures;
    std::map<std::string, Real> node_temperatures;
    std::map<std::string, Real> node_mass_imbalance;
    std::map<std::string, Real> pipe_flow_rates;
    std::map<std::string, Real> pipe_velocities;
    std::map<std::string, Real> pipe_pressure_drops;
    std::map<std::string, Real> pipe_reynolds_numbers;
    std::map<std::string, Real> pipe_friction_factors;
    
    // Quality metrics
    Real max_mass_imbalance = 0.0;
    Real average_iterations_per_pipe = 0.0;
    
    // Helper functions
    Real pressure_drop(const Ptr<Pipe>& pipe) const;
    Real outlet_pressure(const Ptr<Pipe>& pipe) const;
};

// Base solver class
class Solver {
public:
    Solver(Ptr<Network> network, const FluidProperties& fluid);
    virtual ~Solver() = default;
    
    virtual SolutionResults solve();
    virtual void reset();
    
    void set_config(const SolverConfig& config) { config_ = config; }
    const SolverConfig& config() const { return config_; }
    
protected:
    Ptr<Network> network_;
    FluidProperties fluid_;
    SolverConfig config_;
    
    // Cached solutions for warm start
    std::map<std::string, Real> prev_pressures_;
    std::map<std::string, Real> prev_flows_;
};

// Steady-state solver
class SteadyStateSolver : public Solver {
public:
    using Solver::Solver;
    
    SolutionResults solve() override;
    void reset() override;
    
protected:
    // Commercial-grade unified formulation
    bool solve_unified_formulation(int& iterations_performed, Real& final_residual);
    
    bool build_unified_jacobian(
        SparseMatrix& J, Vector& F,
        const Vector& x,
        const std::vector<std::string>& var_names,
        const std::map<std::string, size_t>& var_indices);
    
    void calculate_pipe_flow_advanced(
        const Ptr<Pipe>& pipe, Real p_upstream, Real p_downstream,
        Real& Q, Real& dQ_dp_up, Real& dQ_dp_down, Real& friction_factor);
    
    Real calculate_friction_factor_advanced(
        Real Re, Real relative_roughness, Real& df_dRe);
    
    Real solve_colebrook_white(
        Real Re, Real relative_roughness, Real initial_guess = 0.02);
    
    Real laminar_correction_factor(Real Re, Real& dFactor_dRe);
    
    void update_network_from_solution(
        const Vector& x,
        const std::vector<std::string>& var_names,
        const std::map<std::string, size_t>& var_indices);
    
    void extract_detailed_results(SolutionResults& results);
    
    void enforce_solution_bounds(
        Vector& x, const std::vector<std::string>& var_names);
    
    Real calculate_adaptive_damping(
        Real base_damping, int iteration, Real current_residual, Real prev_residual);
    
    void initialize_solution_advanced(
        Vector& x,
        const std::vector<std::string>& var_names,
        const std::map<std::string, size_t>& var_indices);
    
    // Legacy methods (for compatibility)
    void build_system_matrix(SparseMatrix& A, Vector& b);
    void apply_boundary_conditions(SparseMatrix& A, Vector& b);
    void update_solution(const Vector& x);
    bool check_convergence(const Vector& residual);
    Real calculate_pressure_drop(const Ptr<Pipe>& pipe);
};

} // namespace pipeline_sim

#endif // PIPELINE_SIM_SOLVER_H