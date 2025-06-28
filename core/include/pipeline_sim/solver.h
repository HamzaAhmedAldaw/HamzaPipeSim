#pragma once

#include "pipeline_sim/types.h"
#include "pipeline_sim/network.h"
#include "pipeline_sim/fluid_properties.h"
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>
#include <chrono>
#include <map>
#include <vector>
#include <iostream>

namespace pipeline_sim {

/// Solver configuration
struct SolverConfig {
    Real tolerance{1e-6};
    int max_iterations{100};
    Real relaxation_factor{0.8};
    bool verbose{false};
    bool use_damping{true};
    Real min_damping{0.1};
    Real max_damping{1.0};
    bool check_mass_balance{true};
    Real mass_balance_tolerance{1e-8};
};

/// Solution results
struct SolutionResults {
    bool converged{false};
    int iterations{0};
    Real residual{0.0};
    Real computation_time{0.0};
    Real condition_number{0.0};
    Real mass_imbalance{0.0};
    
    std::map<std::string, Real> node_pressures;
    std::map<std::string, Real> node_temperatures;
    std::map<std::string, Real> pipe_flow_rates;
    std::map<std::string, Real> pipe_pressure_drops;
    std::map<std::string, Real> pipe_velocities;
    
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
    virtual SolutionResults solve() = 0;
    
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
    
    /// Calculate condition number
    Real calculate_condition_number(const SparseMatrix& A);
    
    /// Check mass balance
    Real check_mass_balance();
};

/// Modern Global Newton-Raphson Solver
class GlobalNewtonRaphsonSolver : public Solver {
public:
    using Solver::Solver;
    
    SolutionResults solve() override;
    
protected:
    // System variables
    int num_nodes_;
    int num_pipes_;
    int num_unknowns_;
    int num_fixed_nodes_;
    
    // Node and pipe ordering
    std::vector<std::string> node_order_;
    std::vector<std::string> pipe_order_;
    std::map<std::string, int> node_to_index_;
    std::map<std::string, int> pipe_to_index_;
    
    // Current solution state
    Vector pressures_;
    Vector flows_;
    
    void initialize_system();
    void build_system_matrix(SparseMatrix& A, Vector& b) override;
    void update_solution(const Vector& x) override;
    bool check_convergence(const Vector& residual) override;
    
    // Equation builders
    void add_continuity_equations(std::vector<Eigen::Triplet<Real>>& triplets, 
                                  Vector& b, int& eq_idx);
    void add_energy_equations(std::vector<Eigen::Triplet<Real>>& triplets, 
                              Vector& b, int& eq_idx);
    void add_fixed_grade_equations(std::vector<Eigen::Triplet<Real>>& triplets, 
                                   Vector& b, int& eq_idx);
    
    // Head loss calculations
    Real calculate_head_loss(const Ptr<Pipe>& pipe, Real flow);
    Real calculate_head_loss_derivative(const Ptr<Pipe>& pipe, Real flow);
    
    // Friction factor calculations
    Real calculate_friction_factor(Real reynolds, Real relative_roughness);
    
    // Solution strategies
    bool solve_with_LU(const SparseMatrix& A, const Vector& b, Vector& x);
    bool solve_with_QR(const SparseMatrix& A, const Vector& b, Vector& x);
    bool solve_with_iterative(const SparseMatrix& A, const Vector& b, Vector& x);
    
    // Damping and line search
    Real calculate_damping_factor(const Vector& dx);
    void apply_damping(Vector& dx, Real damping);
};

/// Steady-state solver (wrapper for compatibility)
class SteadyStateSolver : public GlobalNewtonRaphsonSolver {
public:
    using GlobalNewtonRaphsonSolver::GlobalNewtonRaphsonSolver;
};

} // namespace pipeline_sim