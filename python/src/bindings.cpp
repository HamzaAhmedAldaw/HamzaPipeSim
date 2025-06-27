#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include "pipeline_sim/pipeline_sim.h"

namespace py = pybind11;
using namespace pipeline_sim;

PYBIND11_MODULE(_core, m) {
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
        .def("flow_balance", &Node::flow_balance)
        .def("__repr__", [](const Node& n) {
            return "<Node '" + n.id() + "' type=" + 
                   std::to_string(static_cast<int>(n.type())) + ">";
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
        .def("area", &Pipe::area)
        .def("volume", &Pipe::volume)
        .def("velocity", &Pipe::velocity)
        .def("reynolds_number", &Pipe::reynolds_number)
        .def("__repr__", [](const Pipe& p) {
            return "<Pipe '" + p.id() + "' L=" + 
                   std::to_string(p.length()) + "m D=" +
                   std::to_string(p.diameter()) + "m>";
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
        .def_readwrite("api_gravity", &FluidProperties::api_gravity)
        .def("mixture_density", &FluidProperties::mixture_density)
        .def("mixture_viscosity", &FluidProperties::mixture_viscosity);
    
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
        .def_property_readonly("nodes", &Network::nodes)
        .def_property_readonly("pipes", &Network::pipes);
    
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
        .def("pressure_drop", &SolutionResults::pressure_drop)
        .def("outlet_pressure", &SolutionResults::outlet_pressure);
    
    // Solvers
    py::class_<Solver, Ptr<Solver>>(m, "Solver");
    
    py::class_<SteadyStateSolver, Solver, Ptr<SteadyStateSolver>>(m, "SteadyStateSolver")
        .def(py::init<Ptr<Network>, const FluidProperties&>())
        .def("solve", &SteadyStateSolver::solve);
    
    // Module functions
    m.def("get_version", &get_version);
}