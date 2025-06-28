// Stub implementation for transient solver
#include "pipeline_sim/transient_solver.h"

namespace pipeline_sim {

// Minimal implementations to satisfy linker
void ValveClosureEvent::apply(Network& network, Real time) {
    // Stub implementation
}

void PumpTripEvent::apply(Network& network, Real time) {
    // Stub implementation
}

SolutionResults TransientSolver::solve() {
    SolutionResults results;
    results.converged = false;
    return results;
}

void TransientSolver::build_system_matrix(SparseMatrix& A, Vector& b) {
    // Stub implementation
}

void TransientSolver::update_solution(const Vector& x) {
    // Stub implementation
}

bool TransientSolver::check_convergence(const Vector& residual) {
    return false;
}

void TransientSolver::calculate_wave_speeds() {
    // Stub implementation
}

bool TransientSolver::check_cfl_condition() const {
    return true;
}

void TransientSolver::process_events() {
    // Stub implementation
}

void TransientSolver::save_to_history() {
    // Stub implementation
}

void TransientSolver::write_output_header() {
    // Stub implementation
}

void TransientSolver::write_output_state() {
    // Stub implementation
}

void TransientSolver::apply_method_of_characteristics(SparseMatrix& A, Vector& b) {
    // Stub implementation
}

LinePackCalculator::LinePackResult LinePackCalculator::calculate(
    const Network& network,
    const SolutionResults& results,
    const FluidProperties& fluid
) {
    LinePackResult result;
    return result;
}

SurgeAnalyzer::SurgeResult SurgeAnalyzer::analyze(
    const TransientSolver::TimeHistory& history,
    const Network& network,
    Real mawp
) {
    SurgeResult result;
    return result;
}

} // namespace pipeline_sim
