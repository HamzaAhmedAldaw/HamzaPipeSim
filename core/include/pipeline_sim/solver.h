#ifndef PIPELINE_SIM_SOLVER_H
#define PIPELINE_SIM_SOLVER_H

#include <memory>
#include <map>
#include <vector>
#include <string>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "pipeline_sim/types.h"
#include "pipeline_sim/network.h"
#include "pipeline_sim/fluid_properties.h"

namespace pipeline_sim {

// Enhanced solver configuration with industry-standard parameters
struct SolverConfig {
    double tolerance = 1e-6;
    int max_iterations = 100;
    double relaxation_factor = 1.0;
    bool verbose = false;
    
    // Advanced parameters for robust convergence
    bool use_line_search = true;
    double line_search_alpha = 1e-4;  // Armijo constant
    double line_search_beta = 0.5;     // Backtracking factor
    int max_line_search_iter = 20;
    
    bool use_adaptive_relaxation = true;
    double min_relaxation = 0.1;
    double max_relaxation = 1.0;
    
    // Trust region parameters
    bool use_trust_region = false;
    double trust_region_radius = 1e6;  // Pa
    
    // Jacobian calculation method
    enum JacobianMethod {
        FINITE_DIFFERENCE,
        ANALYTICAL
    };
    JacobianMethod jacobian_method = ANALYTICAL;
    double finite_diff_step = 1000.0;  // Pa
    
    // Linear solver options
    enum LinearSolver {
        LU_DECOMPOSITION,
        QR_DECOMPOSITION,
        ITERATIVE_BICGSTAB
    };
    LinearSolver linear_solver = LU_DECOMPOSITION;
    
    // Convergence criteria
    bool check_relative_tolerance = true;
    double relative_tolerance = 1e-8;
    int stagnation_check_window = 5;
    double stagnation_tolerance = 1e-10;
};

// Enhanced solution results with detailed diagnostics
struct SolutionResults {
    bool converged = false;
    int iterations = 0;
    double residual = 0.0;
    double computation_time = 0.0;
    
    std::map<std::string, double> node_pressures;
    std::map<std::string, double> node_temperatures;
    std::map<std::string, double> pipe_flow_rates;
    std::map<std::string, double> pipe_pressure_drops;
    
    // Additional results
    std::map<std::string, double> pipe_velocities;
    std::map<std::string, double> pipe_reynolds_numbers;
    std::map<std::string, double> pipe_friction_factors;
    
    // Convergence history for analysis
    std::vector<double> residual_history;
    std::vector<double> step_size_history;
    
    // Diagnostics
    double jacobian_condition_number = 0.0;
    int jacobian_rank = 0;
    std::string convergence_reason = "";
    
    // Helper methods
    double pressure_drop(const std::shared_ptr<Pipe>& pipe) const;
    double outlet_pressure(const std::shared_ptr<Pipe>& pipe) const;
};

// Base solver class
class Solver {
public:
    Solver(std::shared_ptr<Network> network, const FluidProperties& fluid);
    virtual ~Solver() = default;
    
    virtual SolutionResults solve() = 0;
    
    SolverConfig& config() { return config_; }
    const SolverConfig& config() const { return config_; }
    void set_config(const SolverConfig& cfg) { config_ = cfg; }
    
protected:
    std::shared_ptr<Network> network_;
    FluidProperties fluid_;
    SolverConfig config_;
};

// Professional pipeline solver with industry-standard algorithms
class SteadyStateSolver : public Solver {
public:
    SteadyStateSolver(std::shared_ptr<Network> network, const FluidProperties& fluid);
    
    SolutionResults solve() override;
    
private:
    // System indexing
    std::vector<std::string> unknown_pressure_nodes_;
    std::map<std::string, size_t> node_to_index_;
    std::map<std::string, std::vector<std::string>> node_to_pipes_;
    
    // Private methods
    void setupSolver();
    void printSolverHeader();
    void printSolverSummary(const SolutionResults& results);
    void buildSystemIndexing();
    bool validateNetwork();
    void initializeSolutionProfessional(SolutionResults& results);
    void updatePipeFlows(SolutionResults& results);
    void calculatePipeFlowProfessional(const std::shared_ptr<Pipe>& pipe, SolutionResults& results);
    double calculateFrictionFactorColebrook(double Re, double D, double eps);
    
    void assembleSystemOfEquations(Eigen::SparseMatrix<double>& J, 
                                  Eigen::VectorXd& F,
                                  const SolutionResults& results);
    
    void calculateAnalyticalJacobian(int row, const std::string& node_id,
                                   std::vector<Eigen::Triplet<double>>& triplets,
                                   const SolutionResults& results);
    
    void calculateFiniteDifferenceJacobian(int row, const std::string& node_id,
                                         Eigen::SparseMatrix<double>& J,
                                         const SolutionResults& results,
                                         const Eigen::VectorXd& F_current);
    
    bool solveLinearSystem(const Eigen::SparseMatrix<double>& J,
                          const Eigen::VectorXd& F,
                          Eigen::VectorXd& delta_p,
                          SolutionResults& results);
    
    double performLineSearch(const Eigen::VectorXd& delta_p,
                           const Eigen::VectorXd& F0,
                           double residual0,
                           SolutionResults& results);
    
    double applyTrustRegion(Eigen::VectorXd& delta_p, double current_residual);
    
    double calculateAdaptiveRelaxation(double current_residual, 
                                     double prev_residual,
                                     int iteration);
    
    void updatePressures(const Eigen::VectorXd& delta_p, double step_size,
                       SolutionResults& results);
    
    void evaluateResidual(Eigen::VectorXd& F, const SolutionResults& results);
    
    bool checkConvergence(double current_residual, double prev_residual,
                        const std::vector<double>& recent_residuals, int iteration);
    
    void calculateFinalResults(SolutionResults& results);
};

// Transient solver (stub for now)
class TransientSolver : public Solver {
public:
    TransientSolver(std::shared_ptr<Network> network, const FluidProperties& fluid);
    
    SolutionResults solve() override;
    
    void set_time_step(double dt) { time_step_ = dt; }
    void set_simulation_time(double t) { simulation_time_ = t; }
    void set_output_interval(double interval) { output_interval_ = interval; }
    std::vector<SolutionResults> get_time_history() const { return time_history_; }
    
private:
    double time_step_ = 0.1;
    double simulation_time_ = 100.0;
    double output_interval_ = 1.0;
    std::vector<SolutionResults> time_history_;
};

} // namespace pipeline_sim

#endif // PIPELINE_SIM_SOLVER_H