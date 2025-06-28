#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include "pipeline_sim/pipeline_sim.h"

namespace py = pybind11;
using namespace pipeline_sim;

PYBIND11_MODULE(pipeline_sim, m) {
    m.doc() = "Pipeline-Sim C++ core bindings";
    
    // ===== Enums =====
    py::enum_<NodeType>(m, "NodeType")
        .value("JUNCTION", NodeType::JUNCTION)
        .value("SOURCE", NodeType::SOURCE)
        .value("SINK", NodeType::SINK)
        .value("PUMP", NodeType::PUMP)
        .value("COMPRESSOR", NodeType::COMPRESSOR)
        .value("VALVE", NodeType::VALVE)
        .value("SEPARATOR", NodeType::SEPARATOR)
        .value("HEAT_EXCHANGER", NodeType::HEAT_EXCHANGER);
    
    // ===== Node Class =====
    py::class_<Node, Ptr<Node>>(m, "Node")
        .def(py::init<const std::string&, NodeType>(), 
             py::arg("id"), py::arg("type") = NodeType::JUNCTION)
        // Basic properties
        .def_property_readonly("id", &Node::id)
        .def_property_readonly("type", &Node::type)
        .def_property("pressure", &Node::pressure, &Node::set_pressure)
        .def_property("temperature", &Node::temperature, &Node::set_temperature)
        .def_property("elevation", &Node::elevation, &Node::set_elevation)
        .def("set_type", &Node::set_type)
        // Boundary conditions
        .def("has_pressure_bc", &Node::has_pressure_bc)
        .def("pressure_bc", &Node::pressure_bc)
        .def("set_pressure_bc", &Node::set_pressure_bc)
        .def("remove_pressure_bc", &Node::remove_pressure_bc)
        // Fixed flow
        .def_property("fixed_flow_rate", &Node::fixed_flow_rate, &Node::set_fixed_flow_rate)
        // Pump properties
        .def_property("pump_speed", &Node::pump_speed, &Node::set_pump_speed)
        .def("pump_curve_coefficient_a", &Node::pump_curve_coefficient_a)
        .def("pump_curve_coefficient_b", &Node::pump_curve_coefficient_b)
        .def("set_pump_curve", &Node::set_pump_curve, py::arg("a"), py::arg("b"))
        // Compressor properties
        .def_property("compressor_ratio", &Node::compressor_ratio, &Node::set_compressor_ratio)
        // String representation
        .def("__repr__", [](const Node& n) {
            return "<Node '" + n.id() + "' type=" + 
                   std::to_string(static_cast<int>(n.type())) + 
                   " P=" + std::to_string(n.pressure()) + " Pa>";
        });
    
    // ===== Pipe Class =====
    py::class_<Pipe, Ptr<Pipe>>(m, "Pipe")
        .def(py::init<const std::string&, Ptr<Node>, Ptr<Node>, Real, Real, Real>(),
             py::arg("id"), py::arg("upstream"), py::arg("downstream"),
             py::arg("length"), py::arg("diameter"), py::arg("roughness") = 0.000045)
        // Basic properties
        .def_property_readonly("id", &Pipe::id)
        .def_property_readonly("upstream", &Pipe::upstream)
        .def_property_readonly("downstream", &Pipe::downstream)
        .def_property_readonly("length", &Pipe::length)
        .def_property_readonly("diameter", &Pipe::diameter)
        .def_property("roughness", &Pipe::roughness, &Pipe::set_roughness)
        .def_property("inclination", &Pipe::inclination, &Pipe::set_inclination)
        // Flow properties
        .def_property("flow_rate", &Pipe::flow_rate, &Pipe::set_flow_rate)
        .def_property("velocity", &Pipe::velocity, &Pipe::set_velocity)
        // Calculated properties
        .def("area", &Pipe::area)
        .def("volume", &Pipe::volume)
        .def("reynolds_number", &Pipe::reynolds_number, py::arg("density"), py::arg("viscosity"))
        .def("friction_factor", &Pipe::friction_factor, py::arg("reynolds"))
        // Heat transfer
        .def_property("wall_temperature", &Pipe::wall_temperature, &Pipe::set_wall_temperature)
        .def_property("heat_transfer_coefficient", &Pipe::heat_transfer_coefficient, &Pipe::set_heat_transfer_coefficient)
        // Flow boundary conditions
        .def("has_flow_bc", &Pipe::has_flow_bc)
        .def("flow_bc", &Pipe::flow_bc)
        .def("set_flow_bc", &Pipe::set_flow_bc)
        .def("remove_flow_bc", &Pipe::remove_flow_bc)
        // Valve functionality
        .def("has_valve", &Pipe::has_valve)
        .def("valve_id", &Pipe::valve_id)
        .def("valve_opening", &Pipe::valve_opening)
        .def("set_valve", &Pipe::set_valve, py::arg("valve_id"), py::arg("opening") = 1.0)
        .def("set_valve_opening", &Pipe::set_valve_opening)
        .def("remove_valve", &Pipe::remove_valve)
        // String representation
        .def("__repr__", [](const Pipe& p) {
            return "<Pipe '" + p.id() + "' L=" + 
                   std::to_string(p.length()) + "m D=" +
                   std::to_string(p.diameter()) + "m Q=" +
                   std::to_string(p.flow_rate()) + "m³/s>";
        });
    
    // ===== FluidProperties Class =====
    py::class_<FluidProperties>(m, "FluidProperties")
        .def(py::init<>())
        // Phase densities
        .def_readwrite("oil_density", &FluidProperties::oil_density)
        .def_readwrite("gas_density", &FluidProperties::gas_density)
        .def_readwrite("water_density", &FluidProperties::water_density)
        // Phase viscosities
        .def_readwrite("oil_viscosity", &FluidProperties::oil_viscosity)
        .def_readwrite("gas_viscosity", &FluidProperties::gas_viscosity)
        .def_readwrite("water_viscosity", &FluidProperties::water_viscosity)
        // Phase fractions
        .def_readwrite("oil_fraction", &FluidProperties::oil_fraction)
        .def_readwrite("gas_fraction", &FluidProperties::gas_fraction)
        .def_readwrite("water_fraction", &FluidProperties::water_fraction)
        // Surface tension
        .def_readwrite("oil_water_tension", &FluidProperties::oil_water_tension)
        .def_readwrite("oil_gas_tension", &FluidProperties::oil_gas_tension)
        // Temperature and pressure
        .def_readwrite("temperature", &FluidProperties::temperature)
        .def_readwrite("pressure", &FluidProperties::pressure)
        // PVT properties
        .def_readwrite("gas_oil_ratio", &FluidProperties::gas_oil_ratio)
        .def_readwrite("water_cut", &FluidProperties::water_cut)
        .def_readwrite("bubble_point_pressure", &FluidProperties::bubble_point_pressure)
        .def_readwrite("oil_formation_volume_factor", &FluidProperties::oil_formation_volume_factor)
        .def_readwrite("gas_formation_volume_factor", &FluidProperties::gas_formation_volume_factor)
        .def_readwrite("water_formation_volume_factor", &FluidProperties::water_formation_volume_factor)
        // Phase presence flags
        .def_readwrite("has_oil", &FluidProperties::has_oil)
        .def_readwrite("has_gas", &FluidProperties::has_gas)
        .def_readwrite("has_water", &FluidProperties::has_water)
        // Methods
        .def("is_multiphase", &FluidProperties::is_multiphase)
        .def("mixture_density", &FluidProperties::mixture_density)
        .def("mixture_viscosity", &FluidProperties::mixture_viscosity)
        .def("liquid_fraction", &FluidProperties::liquid_fraction)
        .def("update_pvt", &FluidProperties::update_pvt, py::arg("pressure"))
        .def("gas_z_factor", &FluidProperties::gas_z_factor, py::arg("pressure"), py::arg("temperature"))
        .def("oil_viscosity_at_pressure", &FluidProperties::oil_viscosity_at_pressure, py::arg("pressure"));
    
    // ===== Network Class =====
    py::class_<Network, Ptr<Network>>(m, "Network")
        .def(py::init<>())
        // Node management
        .def("add_node", &Network::add_node, py::arg("id"), py::arg("type"))
        .def("get_node", &Network::get_node, py::arg("id"))
        .def_property_readonly("nodes", &Network::nodes)
        // Pipe management
        .def("add_pipe", &Network::add_pipe, 
             py::arg("id"), py::arg("upstream"), py::arg("downstream"),
             py::arg("length"), py::arg("diameter"))
        .def("get_pipe", &Network::get_pipe, py::arg("id"))
        .def_property_readonly("pipes", &Network::pipes)
        // Connectivity
        .def("get_upstream_pipes", &Network::get_upstream_pipes, py::arg("node"))
        .def("get_downstream_pipes", &Network::get_downstream_pipes, py::arg("node"))
        // Boundary conditions
        .def("set_pressure", &Network::set_pressure, py::arg("node"), py::arg("pressure"))
        .def("set_flow_rate", &Network::set_flow_rate, py::arg("node"), py::arg("flow_rate"))
        .def_property_readonly("pressure_specs", &Network::pressure_specs)
        .def_property_readonly("flow_specs", &Network::flow_specs)
        // Indexing
        .def("node_index", &Network::node_index, py::arg("node_id"))
        .def("pipe_index", &Network::pipe_index, py::arg("pipe_id"))
        // Serialization
        .def("load_from_json", &Network::load_from_json, py::arg("filename"))
        .def("save_to_json", &Network::save_to_json, py::arg("filename"))
        // Network properties
        .def("node_count", &Network::node_count)
        .def("pipe_count", &Network::pipe_count)
        .def("is_valid", &Network::is_valid)
        .def("clear", &Network::clear)
        // String representation
        .def("__repr__", [](const Network& n) {
            return "<Network nodes=" + std::to_string(n.node_count()) + 
                   " pipes=" + std::to_string(n.pipe_count()) + ">";
        });
    
    // ===== SolverConfig Class =====
    py::class_<SolverConfig>(m, "SolverConfig")
        .def(py::init<>())
        .def_readwrite("tolerance", &SolverConfig::tolerance)
        .def_readwrite("relaxation_factor", &SolverConfig::relaxation_factor)
        .def_readwrite("max_iterations", &SolverConfig::max_iterations)
        .def_readwrite("verbose", &SolverConfig::verbose)
        .def_readwrite("pressure_damping", &SolverConfig::pressure_damping)
        .def_readwrite("flow_damping", &SolverConfig::flow_damping)
        .def_readwrite("adaptive_damping", &SolverConfig::adaptive_damping)
        .def_readwrite("use_previous_solution", &SolverConfig::use_previous_solution)
        .def_readwrite("min_pressure", &SolverConfig::min_pressure)
        .def_readwrite("max_pressure", &SolverConfig::max_pressure)
        .def_readwrite("min_flow_velocity", &SolverConfig::min_flow_velocity)
        .def_readwrite("jacobian_epsilon", &SolverConfig::jacobian_epsilon)
        .def_readwrite("laminar_transition_Re", &SolverConfig::laminar_transition_Re)
        .def_readwrite("critical_zone_factor", &SolverConfig::critical_zone_factor)
        .def_readwrite("enable_laminar_correction", &SolverConfig::enable_laminar_correction);
    
    // ===== SolutionResults Class =====
    py::class_<SolutionResults>(m, "SolutionResults")
        .def_readonly("converged", &SolutionResults::converged)
        .def_readonly("iterations", &SolutionResults::iterations)
        .def_readonly("residual", &SolutionResults::residual)
        .def_readonly("computation_time", &SolutionResults::computation_time)
        .def_readonly("node_pressures", &SolutionResults::node_pressures)
        .def_readonly("node_temperatures", &SolutionResults::node_temperatures)
        .def_readonly("node_mass_imbalance", &SolutionResults::node_mass_imbalance)
        .def_readonly("pipe_flow_rates", &SolutionResults::pipe_flow_rates)
        .def_readonly("pipe_velocities", &SolutionResults::pipe_velocities)
        .def_readonly("pipe_pressure_drops", &SolutionResults::pipe_pressure_drops)
        .def_readonly("pipe_reynolds_numbers", &SolutionResults::pipe_reynolds_numbers)
        .def_readonly("pipe_friction_factors", &SolutionResults::pipe_friction_factors)
        .def_readonly("max_mass_imbalance", &SolutionResults::max_mass_imbalance)
        .def_readonly("average_iterations_per_pipe", &SolutionResults::average_iterations_per_pipe)
        .def("pressure_drop", &SolutionResults::pressure_drop, py::arg("pipe"))
        .def("outlet_pressure", &SolutionResults::outlet_pressure, py::arg("pipe"));
    
    // ===== Solver Base Class =====
    py::class_<Solver, Ptr<Solver>>(m, "Solver")
        .def("solve", &Solver::solve)
        .def("reset", &Solver::reset)
        .def("set_config", &Solver::set_config, py::arg("config"))
        .def_property_readonly("config", 
            [](const Solver& s) { return s.config(); },
            py::return_value_policy::copy);
    
    // ===== SteadyStateSolver Class =====
    py::class_<SteadyStateSolver, Solver, Ptr<SteadyStateSolver>>(m, "SteadyStateSolver")
        .def(py::init<Ptr<Network>, const FluidProperties&>(),
             py::arg("network"), py::arg("fluid"))
        .def("solve", &SteadyStateSolver::solve);
    
    // ===== Module-level Functions =====
    m.def("get_version", []() { return std::string("1.0.0"); },
          "Get the version of the pipeline_sim library");
    
    // ===== Physical Constants =====
    py::module_ constants = m.def_submodule("constants", "Physical constants");
    constants.attr("STANDARD_PRESSURE") = py::float_(constants::STANDARD_PRESSURE);
    constants.attr("STANDARD_TEMPERATURE") = py::float_(constants::STANDARD_TEMPERATURE);
    constants.attr("GRAVITY") = py::float_(constants::GRAVITY);
    constants.attr("GAS_CONSTANT") = py::float_(constants::GAS_CONSTANT);
}
