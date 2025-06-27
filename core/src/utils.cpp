// ===== utils.cpp =====
#include "pipeline_sim/utils.h"
#include <fstream>
#include <iomanip>

namespace pipeline_sim {

void save_results_to_csv(const SolutionResults& results, 
                         const std::string& filename) {
    std::ofstream file(filename);
    
    // Write node results
    file << "Node Results\n";
    file << "Node ID,Pressure (Pa),Temperature (K)\n";
    
    for (const auto& [id, pressure] : results.node_pressures) {
        Real temp = results.node_temperatures.at(id);
        file << id << "," << pressure << "," << temp << "\n";
    }
    
    // Write pipe results
    file << "\nPipe Results\n";
    file << "Pipe ID,Flow Rate (m3/s),Pressure Drop (Pa),Liquid Holdup\n";
    
    for (const auto& [id, flow] : results.pipe_flow_rates) {
        Real dp = results.pipe_pressure_drops.at(id);
        Real holdup = 0.0;  // Default if not calculated
        if (results.pipe_liquid_holdups.count(id) > 0) {
            holdup = results.pipe_liquid_holdups.at(id);
        }
        file << id << "," << flow << "," << dp << "," << holdup << "\n";
    }
    
    file.close();
}

void print_results_summary(const SolutionResults& results) {
    std::cout << "\n=== Simulation Results ===\n";
    std::cout << "Converged: " << (results.converged ? "Yes" : "No") << "\n";
    std::cout << "Iterations: " << results.iterations << "\n";
    std::cout << "Final Residual: " << std::scientific 
              << std::setprecision(3) << results.residual << "\n";
    std::cout << "Computation Time: " << std::fixed 
              << std::setprecision(3) << results.computation_time << " s\n";
    
    std::cout << "\nNode Pressures:\n";
    for (const auto& [id, pressure] : results.node_pressures) {
        std::cout << "  " << id << ": " 
                  << pressure / 1e5 << " bar\n";  // Convert to bar
    }
    
    std::cout << "\nPipe Flow Rates:\n";
    for (const auto& [id, flow] : results.pipe_flow_rates) {
        std::cout << "  " << id << ": " 
                  << flow << " m³/s\n";
    }
}

} // namespace pipeline_sim