#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include "pipeline_sim/pipeline_sim.h"

namespace py = pybind11;
using namespace pipeline_sim;

PYBIND11_MODULE(pipeline_sim, m) {
    m.doc() = "Pipeline-Sim C++ core bindings";
    
    // Enums
    py::enum_<NodeType>(m, "NodeType")
        .value("SOURCE", NodeType::SOURCE)
        .value("SINK", NodeType::SINK)
        .value("JUNCTION", NodeType::JUNCTION)
        .value("PUMP", NodeType::PUMP)
        .value("COMPRESSOR", NodeType::COMPRESSOR)
        .value("VALVE", NodeType::VALVE)
        .value("SEPARATOR", NodeType::SEPARATOR);
    
    // Node class
    py::class_<Node, Ptr<Node>>(m, "Node")
        .def(py::init<const std::string&, NodeType>())
        .def_property_readonly("id", &Node::id)
        .def_property_readonly("type", &Node::type)
        .def_property("pressure", &Node::pressure, &Node::set_pressure)
        .def_property("temperature", &Node::temperature, &Node::set_temperature)
        .def_property("elevation", &Node::elevation, &Node::set_elevation)
        .def("has_pressure_bc", &Node::has_pressure_bc)
        .def("pressure_bc", &Node::pressure_bc)
        .def("set_pressure_bc", &Node::set_pressure_bc)
        .def("remove_pressure_bc", &Node::remove_pressure_bc)
        .def("fixed_flow_rate", &Node::fixed_flow_rate)
        .def("set_fixed_flow_rate", &Node::set_fixed_flow_rate)
        .def("__repr__", [](const Node& n) {
            return "<Node '" + n.id() + "' type=" + 
                   std::to_string(static_cast<int>(n.type())) + 
                   " P=" + std::to_string(n.pressure()/1e5) + " bar>";
        });
    
    // Pipe class
    py::class_<Pipe, Ptr<Pipe>>(m, "Pipe")
        .def(py::init<const std::string&, Ptr<Node>, Ptr<Node>, Real, Real>())
        .def_property_readonly("id", &Pipe::id)
        .def_property_readonly("upstream", &Pipe::upstream)
        .def_property_readonly("downstream", &Pipe::downstream)
        .def_property_readonly("length", &Pipe::length)
        .def_property_readonly("diameter", &Pipe::diameter)
        .def_property("roughness", &Pipe::roughness, &Pipe::set_roughness)
        .def_property("inclination", &Pipe::inclination, &Pipe::set_inclination)
        .def_property("flow_rate", &Pipe::flow_rate, &Pipe::set_flow_rate)
        .def_property("velocity", &Pipe::velocity, &Pipe::set_velocity)
        .def("area", &Pipe::area)
        .def("volume", &Pipe::volume)
        .def("reynolds_number", &Pipe::reynolds_number)
        .def("friction_factor", &Pipe::friction_factor)
        .def("__repr__", [](const Pipe& p) {
            return "<Pipe '" + p.id() + "' L=" + 
                   std::to_string(p.length()) + "m D=" +
                   std::to_string(p.diameter()) + "m Q=" +
                   std::to_string(p.flow_rate()) + " m³/s>";
        });
    
    // FluidProperties
    py::class_<FluidProperties>(m, "FluidProperties")
        .def(py::init<>())
        .def_readwrite("oil_density", &FluidProperties::oil_density)
        .def_readwrite("gas_density", &FluidProperties::gas_density)
        .def_readwrite("water_density", &FluidProperties::water_density)
        .def_readwrite("oil_viscosity", &FluidProperties::oil_viscosity)
        .def_readwrite("gas_viscosity", &FluidProperties::gas_viscosity)
        .def_readwrite("water_viscosity", &FluidProperties::water_viscosity)
        .def_readwrite("oil_fraction", &FluidProperties::oil_fraction)
        .def_readwrite("gas_fraction", &FluidProperties::gas_fraction)
        .def_readwrite("water_fraction", &FluidProperties::water_fraction)
        .def_readwrite("gas_oil_ratio", &FluidProperties::gas_oil_ratio)
        .def_readwrite("water_cut", &FluidProperties::water_cut)
        .def_readwrite("temperature", &FluidProperties::temperature)
        .def_readwrite("pressure", &FluidProperties::pressure)
        .def_readwrite("bubble_point_pressure", &FluidProperties::bubble_point_pressure)
        .def_readwrite("oil_formation_volume_factor", &FluidProperties::oil_formation_volume_factor)
        .def_readwrite("gas_formation_volume_factor", &FluidProperties::gas_formation_volume_factor)
        .def_readwrite("water_formation_volume_factor", &FluidProperties::water_formation_volume_factor)
        .def_readwrite("has_oil", &FluidProperties::has_oil)
        .def_readwrite("has_gas", &FluidProperties::has_gas)
        .def_readwrite("has_water", &FluidProperties::has_water)
        .def("mixture_density", &FluidProperties::mixture_density)
        .def("mixture_viscosity", &FluidProperties::mixture_viscosity)
        .def("is_multiphase", &FluidProperties::is_multiphase)
        .def("liquid_fraction", &FluidProperties::liquid_fraction)
        .def("__repr__", [](const FluidProperties& f) {
            return "<FluidProperties density=" + 
                   std::to_string(f.mixture_density()) + " kg/m³ viscosity=" +
                   std::to_string(f.mixture_viscosity()*1000) + " cP>";
        });
    
    // Network
    py::class_<Network, Ptr<Network>>(m, "Network")
        .def(py::init<>())
        .def("add_node", &Network::add_node)
        .def("add_pipe", &Network::add_pipe)
        .def("get_node", &Network::get_node)
        .def("get_pipe", &Network::get_pipe)
        .def("set_pressure", &Network::set_pressure)
        .def("set_flow_rate", &Network::set_flow_rate)
        .def("load_from_json", &Network::load_from_json)
        .def("save_to_json", &Network::save_to_json)
        .def("node_count", &Network::node_count)
        .def("pipe_count", &Network::pipe_count)
        .def("is_valid", &Network::is_valid)
        .def("clear", &Network::clear)
        .def_property_readonly("nodes", &Network::nodes)
        .def_property_readonly("pipes", &Network::pipes)
        .def_property_readonly("pressure_specs", &Network::pressure_specs)
        .def_property_readonly("flow_specs", &Network::flow_specs)
        .def("get_upstream_pipes", &Network::get_upstream_pipes)
        .def("get_downstream_pipes", &Network::get_downstream_pipes)
        .def("node_index", &Network::node_index)
        .def("pipe_index", &Network::pipe_index)
        .def("__repr__", [](const Network& n) {
            return "<Network nodes=" + std::to_string(n.node_count()) + 
                   " pipes=" + std::to_string(n.pipe_count()) + ">";
        });
    
    // SolverConfig
    py::class_<SolverConfig>(m, "SolverConfig")
        .def(py::init<>())
        .def_readwrite("tolerance", &SolverConfig::tolerance)
        .def_readwrite("max_iterations", &SolverConfig::max_iterations)
        .def_readwrite("relaxation_factor", &SolverConfig::relaxation_factor)
        .def_readwrite("verbose", &SolverConfig::verbose)
        .def_readwrite("min_flow_velocity", &SolverConfig::min_flow_velocity)
        .def_readwrite("pressure_damping", &SolverConfig::pressure_damping)
        .def("__repr__", [](const SolverConfig& cfg) {
            return "<SolverConfig tol=" + std::to_string(cfg.tolerance) + 
                   " max_iter=" + std::to_string(cfg.max_iterations) + 
                   " verbose=" + (cfg.verbose ? "True" : "False") + ">";
        });
    
    // SolutionResults
    py::class_<SolutionResults>(m, "SolutionResults")
        .def_readonly("converged", &SolutionResults::converged)
        .def_readonly("iterations", &SolutionResults::iterations)
        .def_readonly("residual", &SolutionResults::residual)
        .def_readonly("computation_time", &SolutionResults::computation_time)
        .def_readonly("node_pressures", &SolutionResults::node_pressures)
        .def_readonly("node_temperatures", &SolutionResults::node_temperatures)
        .def_readonly("pipe_flow_rates", &SolutionResults::pipe_flow_rates)
        .def_readonly("pipe_pressure_drops", &SolutionResults::pipe_pressure_drops)
        .def_readonly("pipe_velocities", &SolutionResults::pipe_velocities)
        .def_readonly("pipe_reynolds_numbers", &SolutionResults::pipe_reynolds_numbers)
        .def("pressure_drop", &SolutionResults::pressure_drop)
        .def("outlet_pressure", &SolutionResults::outlet_pressure)
        .def("__repr__", [](const SolutionResults& r) {
            return "<SolutionResults converged=" + std::string(r.converged ? "True" : "False") + 
                   " iterations=" + std::to_string(r.iterations) + 
                   " residual=" + std::to_string(r.residual) + ">";
        });
    
    // Base Solver class
    py::class_<Solver, Ptr<Solver>>(m, "Solver")
        .def("solve", &Solver::solve)
        .def_property("config",
                      [](Solver& self) -> SolverConfig& { return self.config(); },
                      [](Solver& self, const SolverConfig& cfg) { self.config() = cfg; });
    
    // SteadyStateSolver
    py::class_<SteadyStateSolver, Solver, Ptr<SteadyStateSolver>>(m, "SteadyStateSolver")
        .def(py::init<Ptr<Network>, const FluidProperties&>())
        .def("solve", &SteadyStateSolver::solve)
        .def("__repr__", [](const SteadyStateSolver& s) {
            return "<SteadyStateSolver tolerance=" + 
                   std::to_string(s.config().tolerance) + ">";
        });
    
    // Utility functions
    m.def("get_version", &get_version);
    
    // Constants (if needed)
    py::module constants = m.def_submodule("constants", "Physical constants");
    constants.attr("GRAVITY") = constants::GRAVITY;
    constants.attr("STANDARD_PRESSURE") = constants::STANDARD_PRESSURE;
    constants.attr("STANDARD_TEMPERATURE") = constants::STANDARD_TEMPERATURE;
}
