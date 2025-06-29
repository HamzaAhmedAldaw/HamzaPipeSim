#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <map>
#include <string>
#include <memory>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <vector>
#include <chrono>

// Define constants for Windows
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Include only the headers that work
#include "pipeline_sim/types.h"
#include "pipeline_sim/node.h"
#include "pipeline_sim/pipe.h"
#include "pipeline_sim/fluid_properties.h"
#include "pipeline_sim/network.h"

// Define solver types inline with full implementation
namespace pipeline_sim {
    
    struct SolverConfig {
        double tolerance = 1e-6;
        int max_iterations = 100;
        double relaxation_factor = 1.0;
        bool verbose = false;
    };
    
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
    };
    
    class PipelineSolver {
    public:
        PipelineSolver(std::shared_ptr<Network> network, const FluidProperties& fluid) 
            : network_(network), fluid_(fluid) {}
        
        SolutionResults solve() {
            auto start_time = std::chrono::high_resolution_clock::now();
            SolutionResults results;
            
            if (!network_) {
                results.converged = false;
                return results;
            }
            
            // Initialize with boundary conditions
            initializeSolution(results);
            
            // Get network topology
            auto nodes = network_->nodes();
            auto pipes = network_->pipes();
            
            if (config.verbose) {
                printf("Starting Pipeline Solver:\n");
                printf("  Nodes: %zu\n", nodes.size());
                printf("  Pipes: %zu\n", pipes.size());
                printf("  Tolerance: %.2e\n", config.tolerance);
            }
            
            // Main solver loop - Newton-Raphson method
            for (int iter = 0; iter < config.max_iterations; ++iter) {
                double max_residual = 0.0;
                
                // Build and solve system of equations
                std::map<std::string, double> node_flow_balance;
                
                // Calculate flow rates and pressure drops for each pipe
                for (const auto& [pipe_id, pipe] : pipes) {
                    calculatePipeFlow(pipe, results);
                    
                    // Add to node flow balance
                    double flow = results.pipe_flow_rates[pipe_id];
                    node_flow_balance[pipe->upstream()->id()] -= flow;
                    node_flow_balance[pipe->downstream()->id()] += flow;
                }
                
                // Update pressures for nodes without pressure BC
                for (const auto& [node_id, node] : nodes) {
                    if (node->has_pressure_bc()) {
                        // Fixed pressure - no update needed
                        continue;
                    }
                    
                    // Check flow balance
                    double imbalance = node_flow_balance[node_id];
                    
                    // Add source/sink flows
                    if (node->type() == NodeType::SOURCE || node->type() == NodeType::SINK) {
                        imbalance += node->fixed_flow_rate();
                    }
                    
                    // Update pressure based on flow imbalance
                    double pressure_correction = -imbalance * 1e6; // Simplified
                    double old_pressure = results.node_pressures[node_id];
                    double new_pressure = old_pressure + config.relaxation_factor * pressure_correction;
                    
                    // Apply limits
                    new_pressure = std::max(1e5, std::min(200e5, new_pressure)); // 1-200 bar
                    
                    results.node_pressures[node_id] = new_pressure;
                    
                    double residual = std::abs(imbalance);
                    max_residual = std::max(max_residual, residual);
                }
                
                // Update iteration info
                results.iterations = iter + 1;
                results.residual = max_residual;
                
                if (config.verbose && (iter % 10 == 0 || iter < 5)) {
                    printf("Iteration %d: max residual = %.2e\n", iter + 1, max_residual);
                }
                
                // Check convergence
                if (max_residual < config.tolerance) {
                    results.converged = true;
                    if (config.verbose) {
                        printf("Converged in %d iterations!\n", iter + 1);
                    }
                    break;
                }
            }
            
            // Calculate final results
            calculateFinalResults(results);
            
            // Calculate computation time
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            results.computation_time = duration.count() / 1000.0;
            
            if (config.verbose) {
                printf("Solver completed in %.3f seconds\n", results.computation_time);
            }
            
            return results;
        }
        
        SolverConfig config;
        
    private:
        std::shared_ptr<Network> network_;
        FluidProperties fluid_;
        
        void initializeSolution(SolutionResults& results) {
            // Initialize pressures and temperatures from nodes
            for (const auto& [id, node] : network_->nodes()) {
                results.node_pressures[id] = node->pressure();
                results.node_temperatures[id] = node->temperature();
            }
            
            // Initialize pipe flows (estimate)
            for (const auto& [id, pipe] : network_->pipes()) {
                results.pipe_flow_rates[id] = 0.0;
                results.pipe_pressure_drops[id] = 0.0;
            }
        }
        
        void calculatePipeFlow(const std::shared_ptr<Pipe>& pipe, SolutionResults& results) {
            // Get upstream and downstream pressures
            double p_up = results.node_pressures[pipe->upstream()->id()];
            double p_down = results.node_pressures[pipe->downstream()->id()];
            double dp_friction = 0.0;
            
            // Get pipe properties
            double length = pipe->length();
            double diameter = pipe->diameter();
            double area = M_PI * diameter * diameter / 4.0;
            double roughness = pipe->roughness();
            
            // Get fluid properties
            double density = fluid_.mixture_density();
            double viscosity = fluid_.mixture_viscosity();
            
            // Elevation difference
            double dz = pipe->downstream()->elevation() - pipe->upstream()->elevation();
            double dp_elevation = density * 9.81 * dz;
            
            // Initial guess for flow rate based on pressure difference
            double dp_total = p_up - p_down;
            double dp_friction_available = dp_total - dp_elevation;
            
            // Iterative solution for flow rate (since friction factor depends on velocity)
            double flow_rate = results.pipe_flow_rates[pipe->id()];
            if (flow_rate == 0.0) {
                // Initial estimate
                flow_rate = 0.01 * (dp_friction_available > 0 ? 1 : -1);
            }
            
            // Newton-Raphson iteration for flow rate
            for (int i = 0; i < 10; ++i) {
                double velocity = flow_rate / area;
                double Re = density * std::abs(velocity) * diameter / viscosity;
                
                // Calculate friction factor
                double f = calculateFrictionFactor(Re, diameter, roughness);
                
                // Calculate pressure drop due to friction
                dp_friction = f * length / diameter * 0.5 * density * velocity * std::abs(velocity);
                
                // Total pressure balance
                double dp_calc = dp_friction + dp_elevation;
                double error = dp_total - dp_calc;
                
                // Update flow rate
                double dfdQ = 2.0 * f * length * density * std::abs(velocity) / (diameter * area * area);
                flow_rate += error / dfdQ;
                
                if (std::abs(error) < 100.0) { // 0.001 bar
                    break;
                }
            }
            
            // Store results
            results.pipe_flow_rates[pipe->id()] = flow_rate;
            results.pipe_pressure_drops[pipe->id()] = dp_friction + dp_elevation;
            results.pipe_velocities[pipe->id()] = flow_rate / area;
            results.pipe_reynolds_numbers[pipe->id()] = density * std::abs(flow_rate / area) * diameter / viscosity;
            results.pipe_friction_factors[pipe->id()] = calculateFrictionFactor(
                results.pipe_reynolds_numbers[pipe->id()], diameter, roughness);
        }
        
        double calculateFrictionFactor(double Re, double diameter, double roughness) {
            if (Re < 2300) {
                // Laminar flow
                return 64.0 / Re;
            } else {
                // Turbulent flow - Colebrook-White equation
                double f = 0.02; // Initial guess
                for (int i = 0; i < 5; ++i) {
                    double f_new = 1.0 / std::pow(-2.0 * std::log10(
                        roughness / (3.7 * diameter) + 2.51 / (Re * std::sqrt(f))), 2);
                    if (std::abs(f_new - f) < 1e-6) break;
                    f = f_new;
                }
                return f;
            }
        }
        
        void calculateFinalResults(SolutionResults& results) {
            // Calculate any additional results needed
            double total_pressure_drop = 0.0;
            double total_flow = 0.0;
            
            for (const auto& [id, pipe] : network_->pipes()) {
                total_pressure_drop += std::abs(results.pipe_pressure_drops[id]);
                total_flow += std::abs(results.pipe_flow_rates[id]);
            }
            
            if (config.verbose) {
                printf("\nFinal Results Summary:\n");
                printf("  Total flow: %.3f m³/s (%.0f m³/day)\n", 
                       total_flow / network_->pipes().size(), 
                       total_flow * 86400 / network_->pipes().size());
                printf("  Average pressure drop: %.2f bar\n", 
                       total_pressure_drop / network_->pipes().size() / 1e5);
            }
        }
    };
}

namespace py = pybind11;

PYBIND11_MODULE(pipeline_sim, m) {
    m.doc() = "Pipeline-Sim: Professional Pipeline Simulation";
    
    // NodeType enum
    py::enum_<pipeline_sim::NodeType>(m, "NodeType")
        .value("JUNCTION", pipeline_sim::NodeType::JUNCTION)
        .value("SOURCE", pipeline_sim::NodeType::SOURCE)
        .value("SINK", pipeline_sim::NodeType::SINK)
        .value("PUMP", pipeline_sim::NodeType::PUMP)
        .value("COMPRESSOR", pipeline_sim::NodeType::COMPRESSOR)
        .value("VALVE", pipeline_sim::NodeType::VALVE)
        .value("SEPARATOR", pipeline_sim::NodeType::SEPARATOR)
        .value("HEAT_EXCHANGER", pipeline_sim::NodeType::HEAT_EXCHANGER);
    
    // Node
    py::class_<pipeline_sim::Node, std::shared_ptr<pipeline_sim::Node>>(m, "Node")
        .def(py::init<const std::string&, pipeline_sim::NodeType>())
        .def("id", &pipeline_sim::Node::id)
        .def("type", &pipeline_sim::Node::type)
        .def("pressure", &pipeline_sim::Node::pressure)
        .def("set_pressure", &pipeline_sim::Node::set_pressure)
        .def("temperature", &pipeline_sim::Node::temperature)
        .def("set_temperature", &pipeline_sim::Node::set_temperature)
        .def("elevation", &pipeline_sim::Node::elevation)
        .def("set_elevation", &pipeline_sim::Node::set_elevation)
        .def("has_pressure_bc", &pipeline_sim::Node::has_pressure_bc)
        .def("set_pressure_bc", &pipeline_sim::Node::set_pressure_bc)
        .def("fixed_flow_rate", &pipeline_sim::Node::fixed_flow_rate)
        .def("set_fixed_flow_rate", &pipeline_sim::Node::set_fixed_flow_rate)
        .def("__repr__", [](const pipeline_sim::Node& n) {
            return "<Node '" + n.id() + "' P=" + 
                   std::to_string(n.pressure()/1e5) + " bar>";
        });
    
    // Pipe
    py::class_<pipeline_sim::Pipe, std::shared_ptr<pipeline_sim::Pipe>>(m, "Pipe")
        .def(py::init<const std::string&, 
                      std::shared_ptr<pipeline_sim::Node>, 
                      std::shared_ptr<pipeline_sim::Node>, 
                      double, double>())
        .def("id", &pipeline_sim::Pipe::id)
        .def("upstream", &pipeline_sim::Pipe::upstream)
        .def("downstream", &pipeline_sim::Pipe::downstream)
        .def("length", &pipeline_sim::Pipe::length)
        .def("diameter", &pipeline_sim::Pipe::diameter)
        .def("roughness", &pipeline_sim::Pipe::roughness)
        .def("set_roughness", &pipeline_sim::Pipe::set_roughness)
        .def("inclination", &pipeline_sim::Pipe::inclination)
        .def("set_inclination", &pipeline_sim::Pipe::set_inclination)
        .def("area", &pipeline_sim::Pipe::area)
        .def("volume", &pipeline_sim::Pipe::volume)
        .def("flow_rate", &pipeline_sim::Pipe::flow_rate)
        .def("set_flow_rate", &pipeline_sim::Pipe::set_flow_rate)
        .def("__repr__", [](const pipeline_sim::Pipe& p) {
            return "<Pipe '" + p.id() + "' L=" + 
                   std::to_string(p.length()) + "m D=" + 
                   std::to_string(p.diameter()*39.37) + "\">";
        });
    
    // FluidProperties
    py::class_<pipeline_sim::FluidProperties>(m, "FluidProperties")
        .def(py::init<>())
        .def_readwrite("oil_density", &pipeline_sim::FluidProperties::oil_density)
        .def_readwrite("gas_density", &pipeline_sim::FluidProperties::gas_density)
        .def_readwrite("water_density", &pipeline_sim::FluidProperties::water_density)
        .def_readwrite("oil_viscosity", &pipeline_sim::FluidProperties::oil_viscosity)
        .def_readwrite("gas_viscosity", &pipeline_sim::FluidProperties::gas_viscosity)
        .def_readwrite("water_viscosity", &pipeline_sim::FluidProperties::water_viscosity)
        .def_readwrite("oil_fraction", &pipeline_sim::FluidProperties::oil_fraction)
        .def_readwrite("gas_fraction", &pipeline_sim::FluidProperties::gas_fraction)
        .def_readwrite("water_fraction", &pipeline_sim::FluidProperties::water_fraction)
        .def_readwrite("temperature", &pipeline_sim::FluidProperties::temperature)
        .def_readwrite("pressure", &pipeline_sim::FluidProperties::pressure)
        .def_readwrite("gas_oil_ratio", &pipeline_sim::FluidProperties::gas_oil_ratio)
        .def_readwrite("water_cut", &pipeline_sim::FluidProperties::water_cut)
        .def("mixture_density", &pipeline_sim::FluidProperties::mixture_density)
        .def("mixture_viscosity", &pipeline_sim::FluidProperties::mixture_viscosity)
        .def("is_multiphase", &pipeline_sim::FluidProperties::is_multiphase)
        .def("liquid_fraction", &pipeline_sim::FluidProperties::liquid_fraction);
    
    // Network
    py::class_<pipeline_sim::Network, std::shared_ptr<pipeline_sim::Network>>(m, "Network")
        .def(py::init<>())
        .def("add_node", &pipeline_sim::Network::add_node)
        .def("add_pipe", &pipeline_sim::Network::add_pipe)
        .def("get_node", &pipeline_sim::Network::get_node)
        .def("get_pipe", &pipeline_sim::Network::get_pipe)
        .def("nodes", &pipeline_sim::Network::nodes)
        .def("pipes", &pipeline_sim::Network::pipes)
        .def("set_pressure", 
             static_cast<void (pipeline_sim::Network::*)(const std::shared_ptr<pipeline_sim::Node>&, double)>(&pipeline_sim::Network::set_pressure))
        .def("set_flow_rate", 
             static_cast<void (pipeline_sim::Network::*)(const std::shared_ptr<pipeline_sim::Node>&, double)>(&pipeline_sim::Network::set_flow_rate))
        .def("node_count", &pipeline_sim::Network::node_count)
        .def("pipe_count", &pipeline_sim::Network::pipe_count);
    
    // SolverConfig
    py::class_<pipeline_sim::SolverConfig>(m, "SolverConfig")
        .def(py::init<>())
        .def_readwrite("tolerance", &pipeline_sim::SolverConfig::tolerance)
        .def_readwrite("max_iterations", &pipeline_sim::SolverConfig::max_iterations)
        .def_readwrite("relaxation_factor", &pipeline_sim::SolverConfig::relaxation_factor)
        .def_readwrite("verbose", &pipeline_sim::SolverConfig::verbose);
    
    // SolutionResults
    py::class_<pipeline_sim::SolutionResults>(m, "SolutionResults")
        .def(py::init<>())
        .def_readonly("converged", &pipeline_sim::SolutionResults::converged)
        .def_readonly("iterations", &pipeline_sim::SolutionResults::iterations)
        .def_readonly("residual", &pipeline_sim::SolutionResults::residual)
        .def_readonly("computation_time", &pipeline_sim::SolutionResults::computation_time)
        .def_readonly("node_pressures", &pipeline_sim::SolutionResults::node_pressures)
        .def_readonly("node_temperatures", &pipeline_sim::SolutionResults::node_temperatures)
        .def_readonly("pipe_flow_rates", &pipeline_sim::SolutionResults::pipe_flow_rates)
        .def_readonly("pipe_pressure_drops", &pipeline_sim::SolutionResults::pipe_pressure_drops)
        .def_readonly("pipe_velocities", &pipeline_sim::SolutionResults::pipe_velocities)
        .def_readonly("pipe_reynolds_numbers", &pipeline_sim::SolutionResults::pipe_reynolds_numbers)
        .def_readonly("pipe_friction_factors", &pipeline_sim::SolutionResults::pipe_friction_factors)
        .def("__repr__", [](const pipeline_sim::SolutionResults& r) {
            return "<SolutionResults converged=" + std::string(r.converged ? "True" : "False") + 
                   " iterations=" + std::to_string(r.iterations) + ">";
        });
    
    // PipelineSolver exposed as SteadyStateSolver
    py::class_<pipeline_sim::PipelineSolver>(m, "SteadyStateSolver")
        .def(py::init<std::shared_ptr<pipeline_sim::Network>, const pipeline_sim::FluidProperties&>())
        .def("solve", &pipeline_sim::PipelineSolver::solve)
        .def_readwrite("config", &pipeline_sim::PipelineSolver::config);
    
    // Module attributes
    m.attr("__version__") = "0.1.0";
    m.attr("__author__") = "Pipeline-Sim Team";
    
    // Module-level functions
    m.def("create_example_network", []() {
        auto network = std::make_shared<pipeline_sim::Network>();
        auto inlet = network->add_node("INLET", pipeline_sim::NodeType::SOURCE);
        auto outlet = network->add_node("OUTLET", pipeline_sim::NodeType::SINK);
        auto pipe = network->add_pipe("PIPE1", inlet, outlet, 1000.0, 0.3048);
        network->set_pressure(inlet, 70e5);  // 70 bar
        network->set_flow_rate(outlet, -0.05);  // 0.05 m³/s
        return network;
    }, "Create a simple example network for testing");
    
    m.def("create_example_fluid", []() {
        pipeline_sim::FluidProperties fluid;
        fluid.oil_density = 850;
        fluid.oil_viscosity = 0.002;
        fluid.oil_fraction = 1.0;
        return fluid;
    }, "Create example fluid properties (light oil)");
}
