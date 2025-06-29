#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/stl_bind.h>
#include "pipeline_sim/types.h"
#include "pipeline_sim/node.h"
#include "pipeline_sim/pipe.h"
#include "pipeline_sim/fluid_properties.h"
#include "pipeline_sim/network.h"
#include "pipeline_sim/solver.h"

namespace py = pybind11;

PYBIND11_MODULE(pipeline_sim, m) {
    m.doc() = "Pipeline-Sim: Professional Pipeline Simulation (v2.0)";
    
    // Constants submodule
    auto constants = m.def_submodule("constants", "Physical constants");
    constants.attr("STANDARD_PRESSURE") = py::float_(101325.0);
    constants.attr("GRAVITY") = py::float_(9.81);
    constants.attr("GAS_CONSTANT") = py::float_(8314.0);
    
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
        .def_property_readonly("id", &pipeline_sim::Node::id)
        .def_property_readonly("type", &pipeline_sim::Node::type)
        .def_property("pressure", &pipeline_sim::Node::pressure, &pipeline_sim::Node::set_pressure)
        .def_property("temperature", &pipeline_sim::Node::temperature, &pipeline_sim::Node::set_temperature)
        .def_property("elevation", &pipeline_sim::Node::elevation, &pipeline_sim::Node::set_elevation)
        .def("has_pressure_bc", &pipeline_sim::Node::has_pressure_bc)
        .def("set_pressure_bc", &pipeline_sim::Node::set_pressure_bc)
        .def("pressure_bc", &pipeline_sim::Node::pressure_bc)
        .def("fixed_flow_rate", &pipeline_sim::Node::fixed_flow_rate)
        .def("set_fixed_flow_rate", &pipeline_sim::Node::set_fixed_flow_rate)
        .def("set_flow_rate", &pipeline_sim::Node::set_fixed_flow_rate,
             "Set flow rate for node (alias for set_fixed_flow_rate)")
        .def("pump_speed", &pipeline_sim::Node::pump_speed)
        .def("set_pump_speed", &pipeline_sim::Node::set_pump_speed)
        .def("set_pump_curve", &pipeline_sim::Node::set_pump_curve)
        .def("compressor_ratio", &pipeline_sim::Node::compressor_ratio)
        .def("set_compressor_ratio", &pipeline_sim::Node::set_compressor_ratio)
        .def("__repr__", [](const pipeline_sim::Node& n) {
            return "<Node '" + n.id() + "' P=" + 
                   std::to_string(n.pressure()/1e5) + " bar, z=" +
                   std::to_string(n.elevation()) + " m>";
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
        .def("velocity", &pipeline_sim::Pipe::velocity)
        .def("reynolds_number", [](const pipeline_sim::Pipe& p, double viscosity, double density) {
            return p.reynolds_number(viscosity, density);
        })
        .def("friction_factor", [](const pipeline_sim::Pipe& p, double reynolds) {
            return p.friction_factor(reynolds);
        })
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
        .def("pipe_count", &pipeline_sim::Network::pipe_count)
        .def("is_valid", &pipeline_sim::Network::is_valid)
        .def("clear", &pipeline_sim::Network::clear)
        .def("pressure_specs", &pipeline_sim::Network::pressure_specs)
        .def("flow_specs", &pipeline_sim::Network::flow_specs)
        .def("get_upstream_pipes", &pipeline_sim::Network::get_upstream_pipes)
        .def("get_downstream_pipes", &pipeline_sim::Network::get_downstream_pipes)
        .def("node_index", &pipeline_sim::Network::node_index)
        .def("pipe_index", &pipeline_sim::Network::pipe_index)
        .def("load_from_json", &pipeline_sim::Network::load_from_json)
        .def("save_to_json", &pipeline_sim::Network::save_to_json);
    
    // SolverConfig
    py::class_<pipeline_sim::SolverConfig>(m, "SolverConfig")
        .def(py::init<>())
        .def_readwrite("tolerance", &pipeline_sim::SolverConfig::tolerance)
        .def_readwrite("max_iterations", &pipeline_sim::SolverConfig::max_iterations)
        .def_readwrite("relaxation_factor", &pipeline_sim::SolverConfig::relaxation_factor)
        .def_readwrite("verbose", &pipeline_sim::SolverConfig::verbose)
        .def_readwrite("use_line_search", &pipeline_sim::SolverConfig::use_line_search)
        .def_readwrite("line_search_alpha", &pipeline_sim::SolverConfig::line_search_alpha)
        .def_readwrite("line_search_beta", &pipeline_sim::SolverConfig::line_search_beta)
        .def_readwrite("max_line_search_iter", &pipeline_sim::SolverConfig::max_line_search_iter)
        .def_readwrite("use_adaptive_relaxation", &pipeline_sim::SolverConfig::use_adaptive_relaxation)
        .def_readwrite("min_relaxation", &pipeline_sim::SolverConfig::min_relaxation)
        .def_readwrite("max_relaxation", &pipeline_sim::SolverConfig::max_relaxation)
        .def_readwrite("use_trust_region", &pipeline_sim::SolverConfig::use_trust_region)
        .def_readwrite("trust_region_radius", &pipeline_sim::SolverConfig::trust_region_radius)
        .def_readwrite("check_relative_tolerance", &pipeline_sim::SolverConfig::check_relative_tolerance)
        .def_readwrite("relative_tolerance", &pipeline_sim::SolverConfig::relative_tolerance)
        .def_readwrite("stagnation_check_window", &pipeline_sim::SolverConfig::stagnation_check_window)
        .def_readwrite("stagnation_tolerance", &pipeline_sim::SolverConfig::stagnation_tolerance);
    
    // Jacobian method enum
    py::enum_<pipeline_sim::SolverConfig::JacobianMethod>(m, "JacobianMethod")
        .value("FINITE_DIFFERENCE", pipeline_sim::SolverConfig::FINITE_DIFFERENCE)
        .value("ANALYTICAL", pipeline_sim::SolverConfig::ANALYTICAL);
    
    // Linear solver enum
    py::enum_<pipeline_sim::SolverConfig::LinearSolver>(m, "LinearSolver")
        .value("LU_DECOMPOSITION", pipeline_sim::SolverConfig::LU_DECOMPOSITION)
        .value("QR_DECOMPOSITION", pipeline_sim::SolverConfig::QR_DECOMPOSITION)
        .value("ITERATIVE_BICGSTAB", pipeline_sim::SolverConfig::ITERATIVE_BICGSTAB);
    
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
        .def_readonly("residual_history", &pipeline_sim::SolutionResults::residual_history)
        .def_readonly("step_size_history", &pipeline_sim::SolutionResults::step_size_history)
        .def_readonly("jacobian_condition_number", &pipeline_sim::SolutionResults::jacobian_condition_number)
        .def_readonly("jacobian_rank", &pipeline_sim::SolutionResults::jacobian_rank)
        .def_readonly("convergence_reason", &pipeline_sim::SolutionResults::convergence_reason)
        .def("pressure_drop", &pipeline_sim::SolutionResults::pressure_drop)
        .def("outlet_pressure", &pipeline_sim::SolutionResults::outlet_pressure)
        .def("__repr__", [](const pipeline_sim::SolutionResults& r) {
            return "<SolutionResults converged=" + std::string(r.converged ? "True" : "False") + 
                   " iterations=" + std::to_string(r.iterations) + 
                   " residual=" + std::to_string(r.residual) + ">";
        });
    
    // Base Solver class
    py::class_<pipeline_sim::Solver, std::shared_ptr<pipeline_sim::Solver>>(m, "Solver")
        .def("solve", &pipeline_sim::Solver::solve)
        .def_property("config", 
            (pipeline_sim::SolverConfig& (pipeline_sim::Solver::*)()) &pipeline_sim::Solver::config,
            &pipeline_sim::Solver::set_config,
            py::return_value_policy::reference_internal);
    
    // SteadyStateSolver
    py::class_<pipeline_sim::SteadyStateSolver, pipeline_sim::Solver, std::shared_ptr<pipeline_sim::SteadyStateSolver>>(m, "SteadyStateSolver")
        .def(py::init<std::shared_ptr<pipeline_sim::Network>, const pipeline_sim::FluidProperties&>())
        .def("solve", &pipeline_sim::SteadyStateSolver::solve);
    
    // TransientSolver
    py::class_<pipeline_sim::TransientSolver, pipeline_sim::Solver, std::shared_ptr<pipeline_sim::TransientSolver>>(m, "TransientSolver")
        .def(py::init<std::shared_ptr<pipeline_sim::Network>, const pipeline_sim::FluidProperties&>())
        .def("solve", &pipeline_sim::TransientSolver::solve)
        .def("set_time_step", &pipeline_sim::TransientSolver::set_time_step)
        .def("set_simulation_time", &pipeline_sim::TransientSolver::set_simulation_time)
        .def("set_output_interval", &pipeline_sim::TransientSolver::set_output_interval)
        .def("get_time_history", &pipeline_sim::TransientSolver::get_time_history);
    
    // Module attributes
    m.attr("__version__") = "2.0.0";
    m.attr("__author__") = "Pipeline-Sim Professional Team";
    
    // Module-level functions
    m.def("create_example_network", []() {
        auto network = std::make_shared<pipeline_sim::Network>();
        auto inlet = network->add_node("INLET", pipeline_sim::NodeType::SOURCE);
        auto outlet = network->add_node("OUTLET", pipeline_sim::NodeType::SINK);
        inlet->set_elevation(0.0);
        outlet->set_elevation(0.0);
        auto pipe = network->add_pipe("PIPE1", inlet, outlet, 1000.0, 0.3048);
        pipe->set_roughness(0.000045);
        network->set_pressure(inlet, 70e5);  // 70 bar
        network->set_pressure(outlet, 69e5);  // 69 bar
        return network;
    }, "Create a simple example network for testing");
    
    m.def("create_example_fluid", []() {
        pipeline_sim::FluidProperties fluid;
        fluid.oil_density = 850;
        fluid.oil_viscosity = 0.002;
        fluid.oil_fraction = 1.0;
        fluid.gas_fraction = 0.0;
        fluid.water_fraction = 0.0;
        return fluid;
    }, "Create example fluid properties (light oil)");
    
    m.def("get_version", []() {
        return std::string("2.0.0");
    }, "Get Pipeline-Sim version");
}
