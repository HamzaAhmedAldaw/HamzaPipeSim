#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

#include "pipeline_sim/network.h"
#include "pipeline_sim/node.h"
#include "pipeline_sim/pipe.h"
#include "pipeline_sim/fluid_properties.h"
#include "pipeline_sim/solver.h"
#include "pipeline_sim/transient_solver.h"
#include "pipeline_sim/correlations.h"
#include "pipeline_sim/equipment.h"

namespace py = pybind11;
using namespace pipeline_sim;

PYBIND11_MODULE(pipeline_sim, m) {
    m.doc() = "HamzaPipeSim - Pipeline Network Simulation (Core Features Only)";

    // Enums
    py::enum_<NodeType>(m, "NodeType")
        .value("SOURCE", NodeType::SOURCE)
        .value("SINK", NodeType::SINK)
        .value("JUNCTION", NodeType::JUNCTION);

    py::enum_<FlowPattern>(m, "FlowPattern")
        .value("SINGLE_PHASE", FlowPattern::SINGLE_PHASE)
        .value("SEGREGATED", FlowPattern::SEGREGATED)
        .value("INTERMITTENT", FlowPattern::INTERMITTENT)
        .value("DISTRIBUTED", FlowPattern::DISTRIBUTED);

    // Node class
    py::class_<Node>(m, "Node")
        .def_property_readonly("id", &Node::id)
        .def_property_readonly("name", &Node::name)
        .def_property_readonly("type", &Node::type)
        .def_property("pressure", &Node::pressure, &Node::set_pressure)
        .def_property("flow_rate", &Node::flow_rate, &Node::set_flow_rate);

    // Pipe class
    py::class_<Pipe>(m, "Pipe")
        .def_property_readonly("id", &Pipe::id)
        .def_property_readonly("name", &Pipe::name)
        .def_property_readonly("from_node", &Pipe::from_node)
        .def_property_readonly("to_node", &Pipe::to_node)
        .def_property_readonly("length", &Pipe::length)
        .def_property_readonly("diameter", &Pipe::diameter)
        .def_property_readonly("roughness", &Pipe::roughness)
        .def_property_readonly("elevation_change", &Pipe::elevation_change);

    // FluidProperties
    py::class_<FluidProperties>(m, "FluidProperties")
        .def(py::init<>())
        .def_readwrite("oil_density", &FluidProperties::oil_density)
        .def_readwrite("oil_viscosity", &FluidProperties::oil_viscosity)
        .def_readwrite("gas_density", &FluidProperties::gas_density)
        .def_readwrite("gas_viscosity", &FluidProperties::gas_viscosity)
        .def_readwrite("water_density", &FluidProperties::water_density)
        .def_readwrite("water_viscosity", &FluidProperties::water_viscosity)
        .def_readwrite("gas_oil_ratio", &FluidProperties::gas_oil_ratio)
        .def_readwrite("water_cut", &FluidProperties::water_cut)
        .def_readwrite("temperature", &FluidProperties::temperature)
        .def_readwrite("gas_z_factor", &FluidProperties::gas_z_factor)
        .def_readwrite("gas_specific_gravity", &FluidProperties::gas_specific_gravity);

    // Network
    py::class_<Network>(m, "Network")
        .def(py::init<>())
        .def("add_node", &Network::add_node, py::arg("name"), py::arg("type"))
        .def("add_pipe", &Network::add_pipe,
             py::arg("name"), py::arg("from_node"), py::arg("to_node"),
             py::arg("length"), py::arg("diameter"), py::arg("roughness") = 0.045,
             py::arg("elevation_change") = 0.0)
        .def("set_pressure", &Network::set_pressure)
        .def("set_flow_rate", &Network::set_flow_rate)
        .def("get_node", &Network::get_node, py::return_value_policy::reference)
        .def("get_pipe", &Network::get_pipe, py::return_value_policy::reference)
        .def("node_count", &Network::node_count)
        .def("pipe_count", &Network::pipe_count)
        .def("is_valid", &Network::is_valid)
        .def("clear", &Network::clear);

    // SolverConfig
    py::class_<SolverConfig>(m, "SolverConfig")
        .def(py::init<>())
        .def_readwrite("tolerance", &SolverConfig::tolerance)
        .def_readwrite("max_iterations", &SolverConfig::max_iterations)
        .def_readwrite("relaxation_factor", &SolverConfig::relaxation_factor);

    // SolutionResults
    py::class_<SolutionResults>(m, "SolutionResults")
        .def_readonly("converged", &SolutionResults::converged)
        .def_readonly("iterations", &SolutionResults::iterations)
        .def_readonly("residual", &SolutionResults::residual)
        .def_readonly("node_pressures", &SolutionResults::node_pressures)
        .def_readonly("pipe_flow_rates", &SolutionResults::pipe_flow_rates);

    // SteadyStateSolver
    py::class_<SteadyStateSolver>(m, "SteadyStateSolver")
        .def(py::init<Network&, const FluidProperties&>())
        .def("solve", &SteadyStateSolver::solve)
        .def("config", &SteadyStateSolver::config, py::return_value_policy::reference);

    // TransientSolver
    py::class_<TransientSolver>(m, "TransientSolver")
        .def(py::init<Network&, const FluidProperties&>())
        .def("solve", &TransientSolver::solve)
        .def("config", &TransientSolver::config, py::return_value_policy::reference);

    // Basic correlations
    py::class_<BeggsBrill>(m, "BeggsBrill")
        .def(py::init<>())
        .def("calculate", &BeggsBrill::calculate);

    py::class_<HagedornBrown>(m, "HagedornBrown")
        .def(py::init<>())
        .def("calculate", &HagedornBrown::calculate);
}
