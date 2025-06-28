#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/stl_bind.h>
#include "pipeline_sim/pipeline_sim.h"
#include "pipeline_sim/equipment.h"
#include "pipeline_sim/transient_solver.h"
#include "pipeline_sim/ml_integration.h"

namespace py = pybind11;
using namespace pipeline_sim;

// Make STL containers opaque to avoid conversion overhead
PYBIND11_MAKE_OPAQUE(std::vector<std::string>);
PYBIND11_MAKE_OPAQUE(std::map<std::string, Real>);

PYBIND11_MODULE(pipeline_sim, m) {
    m.doc() = "Pipeline-Sim C++ core bindings";
    
    // Bind STL containers
    py::bind_vector<std::vector<std::string>>(m, "VectorString");
    py::bind_map<std::map<std::string, Real>>(m, "MapStringReal");
    
    // ========== Enums ==========
    
    py::enum_<NodeType>(m, "NodeType")
        .value("JUNCTION", NodeType::JUNCTION)
        .value("SOURCE", NodeType::SOURCE)
        .value("SINK", NodeType::SINK)
        .value("PUMP", NodeType::PUMP)
        .value("COMPRESSOR", NodeType::COMPRESSOR)
        .value("VALVE", NodeType::VALVE)
        .value("SEPARATOR", NodeType::SEPARATOR)
        .value("HEAT_EXCHANGER", NodeType::HEAT_EXCHANGER);
    
    py::enum_<FlowPattern>(m, "FlowPattern")
        .value("SEGREGATED", FlowPattern::SEGREGATED)
        .value("INTERMITTENT", FlowPattern::INTERMITTENT)
        .value("DISTRIBUTED", FlowPattern::DISTRIBUTED)
        .value("ANNULAR", FlowPattern::ANNULAR)
        .value("BUBBLE", FlowPattern::BUBBLE)
        .value("SLUG", FlowPattern::SLUG)
        .value("CHURN", FlowPattern::CHURN)
        .value("MIST", FlowPattern::MIST);
    
    // ========== Core Classes ==========
    
    // Node class
    py::class_<Node, Ptr<Node>>(m, "Node")
        .def(py::init<const std::string&, NodeType>(),
             py::arg("id"), py::arg("type") = NodeType::JUNCTION)
        .def_property_readonly("id", &Node::id)
        .def_property_readonly("type", &Node::type)
        .def_property("pressure", &Node::pressure, &Node::set_pressure)
        .def_property("temperature", &Node::temperature, &Node::set_temperature)
        .def_property("elevation", &Node::elevation, &Node::set_elevation)
        .def("set_type", &Node::set_type)
        .def("has_pressure_bc", &Node::has_pressure_bc)
        .def("pressure_bc", &Node::pressure_bc)
        .def("set_pressure_bc", &Node::set_pressure_bc)
        .def("remove_pressure_bc", &Node::remove_pressure_bc)
        .def("fixed_flow_rate", &Node::fixed_flow_rate)
        .def("set_fixed_flow_rate", &Node::set_fixed_flow_rate)
        .def("pump_speed", &Node::pump_speed)
        .def("set_pump_speed", &Node::set_pump_speed)
        .def("pump_curve_coefficient_a", &Node::pump_curve_coefficient_a)
        .def("pump_curve_coefficient_b", &Node::pump_curve_coefficient_b)
        .def("set_pump_curve", &Node::set_pump_curve)
        .def("compressor_ratio", &Node::compressor_ratio)
        .def("set_compressor_ratio", &Node::set_compressor_ratio)
        .def("__repr__", [](const Node& n) {
            return "<Node '" + n.id() + "' type=" + 
                   std::to_string(static_cast<int>(n.type())) + ">";
        });
    
    // Pipe class
    py::class_<Pipe, Ptr<Pipe>>(m, "Pipe")
        .def(py::init<const std::string&, Ptr<Node>, Ptr<Node>, Real, Real>(),
             py::arg("id"), py::arg("upstream"), py::arg("downstream"),
             py::arg("length"), py::arg("diameter"))
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
        .def("has_valve", &Pipe::has_valve)
        .def("valve_id", &Pipe::valve_id)
        .def("valve_opening", &Pipe::valve_opening)
        .def("set_valve", &Pipe::set_valve, py::arg("valve_id"), py::arg("opening") = 1.0)
        .def("set_valve_opening", &Pipe::set_valve_opening)
        .def("remove_valve", &Pipe::remove_valve)
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
        .def_readwrite("oil_water_tension", &FluidProperties::oil_water_tension)
        .def_readwrite("oil_gas_tension", &FluidProperties::oil_gas_tension)
        .def_readwrite("temperature", &FluidProperties::temperature)
        .def_readwrite("pressure", &FluidProperties::pressure)
        .def_readwrite("gas_oil_ratio", &FluidProperties::gas_oil_ratio)
        .def_readwrite("water_cut", &FluidProperties::water_cut)
        .def_readwrite("bubble_point_pressure", &FluidProperties::bubble_point_pressure)
        .def_readwrite("oil_formation_volume_factor", &FluidProperties::oil_formation_volume_factor)
        .def_readwrite("gas_formation_volume_factor", &FluidProperties::gas_formation_volume_factor)
        .def_readwrite("water_formation_volume_factor", &FluidProperties::water_formation_volume_factor)
        .def_readwrite("has_oil", &FluidProperties::has_oil)
        .def_readwrite("has_gas", &FluidProperties::has_gas)
        .def_readwrite("has_water", &FluidProperties::has_water)
        .def("is_multiphase", &FluidProperties::is_multiphase)
        .def("mixture_density", &FluidProperties::mixture_density)
        .def("mixture_viscosity", &FluidProperties::mixture_viscosity)
        .def("liquid_fraction", &FluidProperties::liquid_fraction)
        .def("update_pvt", &FluidProperties::update_pvt)
        .def("gas_z_factor", &FluidProperties::gas_z_factor)
        .def("oil_viscosity_at_pressure", &FluidProperties::oil_viscosity_at_pressure);
    
    // Network
    py::class_<Network, Ptr<Network>>(m, "Network")
        .def(py::init<>())
        .def("add_node", &Network::add_node)
        .def("add_pipe", &Network::add_pipe)
        .def("get_node", &Network::get_node)
        .def("get_pipe", &Network::get_pipe)
        .def("nodes", &Network::nodes)
        .def("pipes", &Network::pipes)
        .def("get_upstream_pipes", &Network::get_upstream_pipes)
        .def("get_downstream_pipes", &Network::get_downstream_pipes)
        .def("set_pressure", &Network::set_pressure)
        .def("set_flow_rate", &Network::set_flow_rate)
        .def("pressure_specs", &Network::pressure_specs)
        .def("flow_specs", &Network::flow_specs)
        .def("node_index", &Network::node_index)
        .def("pipe_index", &Network::pipe_index)
        .def("load_from_json", &Network::load_from_json)
        .def("save_to_json", &Network::save_to_json)
        .def("node_count", &Network::node_count)
        .def("pipe_count", &Network::pipe_count)
        .def("is_valid", &Network::is_valid)
        .def("clear", &Network::clear);
    
    // ========== Correlation Classes ==========
    
    // FlowCorrelation Results
    py::class_<FlowCorrelation::Results>(m, "FlowCorrelationResults")
        .def(py::init<>())
        .def_readwrite("pressure_gradient", &FlowCorrelation::Results::pressure_gradient)
        .def_readwrite("liquid_holdup", &FlowCorrelation::Results::liquid_holdup)
        .def_readwrite("flow_pattern", &FlowCorrelation::Results::flow_pattern)
        .def_readwrite("friction_factor", &FlowCorrelation::Results::friction_factor)
        .def_readwrite("mixture_density", &FlowCorrelation::Results::mixture_density)
        .def_readwrite("mixture_velocity", &FlowCorrelation::Results::mixture_velocity)
        .def_readwrite("additional_data", &FlowCorrelation::Results::additional_data);
    
    // Base FlowCorrelation class
    py::class_<FlowCorrelation, std::unique_ptr<FlowCorrelation, py::nodelete>>(m, "FlowCorrelation")
        .def("calculate", &FlowCorrelation::calculate)
        .def("name", &FlowCorrelation::name)
        .def("is_applicable", &FlowCorrelation::is_applicable);
    
    // Concrete correlation classes
    py::class_<BeggsBrillCorrelation, FlowCorrelation>(m, "BeggsBrillCorrelation")
        .def(py::init<>());
    
    py::class_<HagedornBrownCorrelation, FlowCorrelation>(m, "HagedornBrownCorrelation")
        .def(py::init<>());
    
    py::class_<GrayCorrelation, FlowCorrelation>(m, "GrayCorrelation")
        .def(py::init<>());
    
    py::class_<MechanisticModel, FlowCorrelation>(m, "MechanisticModel")
        .def(py::init<>());
    
    // CorrelationFactory
    py::class_<CorrelationFactory>(m, "CorrelationFactory")
        .def_static("create", &CorrelationFactory::create)
        .def_static("available_correlations", &CorrelationFactory::available_correlations);
    
    // ========== Equipment Classes ==========
    
    // Base Equipment class
    py::class_<Equipment, Ptr<Equipment>>(m, "Equipment")
        .def("calculate", &Equipment::calculate)
        .def("power_consumption", &Equipment::power_consumption)
        .def("efficiency", &Equipment::efficiency)
        .def_property_readonly("id", &Equipment::id)
        .def_property_readonly("type", &Equipment::type);
    
    // CentrifugalPump
    py::class_<CentrifugalPump, Equipment, Ptr<CentrifugalPump>>(m, "CentrifugalPump")
        .def(py::init<const std::string&>())
        .def("set_curve_coefficients", &CentrifugalPump::set_curve_coefficients)
        .def("set_speed_ratio", &CentrifugalPump::set_speed_ratio)
        .def("set_efficiency_curve", &CentrifugalPump::set_efficiency_curve)
        .def("head", &CentrifugalPump::head);
    
    // Compressor
    py::enum_<Compressor::Type>(m, "CompressorType")
        .value("CENTRIFUGAL", Compressor::Type::CENTRIFUGAL)
        .value("RECIPROCATING", Compressor::Type::RECIPROCATING)
        .value("SCREW", Compressor::Type::SCREW);
    
    py::class_<Compressor, Equipment, Ptr<Compressor>>(m, "Compressor")
        .def(py::init<const std::string&, Compressor::Type>(),
             py::arg("id"), py::arg("type") = Compressor::Type::CENTRIFUGAL)
        .def("set_performance_map", &Compressor::set_performance_map)
        .def("set_polytropic_efficiency", &Compressor::set_polytropic_efficiency);
    
    // ControlValve
    py::enum_<ControlValve::Characteristic>(m, "ValveCharacteristic")
        .value("LINEAR", ControlValve::Characteristic::LINEAR)
        .value("EQUAL_PERCENTAGE", ControlValve::Characteristic::EQUAL_PERCENTAGE)
        .value("QUICK_OPENING", ControlValve::Characteristic::QUICK_OPENING);
    
    py::class_<ControlValve, Equipment, Ptr<ControlValve>>(m, "ControlValve")
        .def(py::init<const std::string&>())
        .def("set_cv", &ControlValve::set_cv)
        .def("set_opening", &ControlValve::set_opening)
        .def("set_characteristic", &ControlValve::set_characteristic)
        .def_static("required_cv", &ControlValve::required_cv);
    
    // Separator
    py::enum_<Separator::Type>(m, "SeparatorType")
        .value("TWO_PHASE", Separator::Type::TWO_PHASE)
        .value("THREE_PHASE", Separator::Type::THREE_PHASE);
    
    py::class_<Separator::SeparatedFlows>(m, "SeparatedFlows")
        .def_readwrite("gas_flow", &Separator::SeparatedFlows::gas_flow)
        .def_readwrite("oil_flow", &Separator::SeparatedFlows::oil_flow)
        .def_readwrite("water_flow", &Separator::SeparatedFlows::water_flow);
    
    py::class_<Separator, Equipment, Ptr<Separator>>(m, "Separator")
        .def(py::init<const std::string&, Separator::Type>(),
             py::arg("id"), py::arg("type") = Separator::Type::TWO_PHASE)
        .def("set_separation_efficiency", &Separator::set_separation_efficiency)
        .def("set_residence_time", &Separator::set_residence_time)
        .def("get_separated_flows", &Separator::get_separated_flows);
    
    // HeatExchanger
    py::enum_<HeatExchanger::Type>(m, "HeatExchangerType")
        .value("SHELL_AND_TUBE", HeatExchanger::Type::SHELL_AND_TUBE)
        .value("PLATE", HeatExchanger::Type::PLATE)
        .value("AIR_COOLED", HeatExchanger::Type::AIR_COOLED);
    
    py::class_<HeatExchanger, Equipment, Ptr<HeatExchanger>>(m, "HeatExchanger")
        .def(py::init<const std::string&, HeatExchanger::Type>(),
             py::arg("id"), py::arg("type") = HeatExchanger::Type::SHELL_AND_TUBE)
        .def("set_ua_value", &HeatExchanger::set_ua_value)
        .def("set_cooling_temperature", &HeatExchanger::set_cooling_temperature)
        .def("heat_duty", &HeatExchanger::heat_duty);
    
    // ========== Solver Classes ==========
    
    // SolverConfig
    py::class_<SolverConfig>(m, "SolverConfig")
        .def(py::init<>())
        .def_readwrite("tolerance", &SolverConfig::tolerance)
        .def_readwrite("max_iterations", &SolverConfig::max_iterations)
        .def_readwrite("relaxation_factor", &SolverConfig::relaxation_factor)
        .def_readwrite("verbose", &SolverConfig::verbose);
    
    // SolutionResults
    py::class_<SolutionResults>(m, "SolutionResults")
        .def(py::init<>())
        .def_readwrite("converged", &SolutionResults::converged)
        .def_readwrite("iterations", &SolutionResults::iterations)
        .def_readwrite("residual", &SolutionResults::residual)
        .def_readwrite("computation_time", &SolutionResults::computation_time)
        .def_readwrite("node_pressures", &SolutionResults::node_pressures)
        .def_readwrite("node_temperatures", &SolutionResults::node_temperatures)
        .def_readwrite("pipe_flow_rates", &SolutionResults::pipe_flow_rates)
        .def_readwrite("pipe_pressure_drops", &SolutionResults::pipe_pressure_drops)
        .def("pressure_drop", &SolutionResults::pressure_drop)
        .def("outlet_pressure", &SolutionResults::outlet_pressure);
    
    // Base Solver class
    py::class_<Solver, Ptr<Solver>>(m, "Solver")
        .def("solve", &Solver::solve)
        .def("config", (SolverConfig& (Solver::*)()) &Solver::config,
             py::return_value_policy::reference_internal);
    
    // SteadyStateSolver
    py::class_<SteadyStateSolver, Solver, Ptr<SteadyStateSolver>>(m, "SteadyStateSolver")
        .def(py::init<Ptr<Network>, const FluidProperties&>())
        .def("solve", &SteadyStateSolver::solve);
    
    // TransientSolver
    py::class_<TransientSolver, Solver, Ptr<TransientSolver>>(m, "TransientSolver")
        .def(py::init<Ptr<Network>, const FluidProperties&>())
        .def("solve", &TransientSolver::solve)
        .def("set_time_step", &TransientSolver::set_time_step)
        .def("set_simulation_time", &TransientSolver::set_simulation_time)
        .def("set_output_interval", &TransientSolver::set_output_interval)
        .def("get_time_history", &TransientSolver::get_time_history);
    
    // ========== PVT Models ==========
    
    // BlackOilModel
    py::class_<BlackOilModel>(m, "BlackOilModel")
        .def_static("oil_fvf", &BlackOilModel::oil_fvf)
        .def_static("gas_fvf", &BlackOilModel::gas_fvf)
        .def_static("solution_gor", &BlackOilModel::solution_gor)
        .def_static("oil_viscosity", &BlackOilModel::oil_viscosity);
    
    // GasProperties
    py::class_<GasProperties>(m, "GasProperties")
        .def_static("pseudo_critical_properties", &GasProperties::pseudo_critical_properties)
        .def_static("z_factor_dpr", &GasProperties::z_factor_dpr)
        .def_static("viscosity_lge", &GasProperties::viscosity_lge);
    
    // WaterProperties
    py::class_<WaterProperties>(m, "WaterProperties")
        .def_static("water_fvf", &WaterProperties::water_fvf)
        .def_static("water_viscosity", &WaterProperties::water_viscosity)
        .def_static("water_density", &WaterProperties::water_density);
    
    // ========== ML Module ==========
    
    auto ml_module = m.def_submodule("ml", "Machine Learning integration");
    
    // FeatureExtractor
    py::class_<ml::FeatureExtractor>(ml_module, "FeatureExtractor")
        .def_static("extract_features", &ml::FeatureExtractor::extract_features)
        .def_static("get_feature_names", &ml::FeatureExtractor::get_feature_names)
        .def_static("normalize_features", &ml::FeatureExtractor::normalize_features);
    
    // Base MLModel class
    py::class_<ml::MLModel, std::shared_ptr<ml::MLModel>>(ml_module, "MLModel")
        .def("load", &ml::MLModel::load)
        .def("predict", &ml::MLModel::predict)
        .def("info", &ml::MLModel::info);
    
    // FlowPatternPredictor
    py::class_<ml::FlowPatternPredictor, ml::MLModel, std::shared_ptr<ml::FlowPatternPredictor>>(ml_module, "FlowPatternPredictor")
        .def(py::init<>())
        .def("predict_pattern", &ml::FlowPatternPredictor::predict_pattern);
    
    // AnomalyDetector
    py::class_<ml::AnomalyDetector::AnomalyResult>(ml_module, "AnomalyResult")
        .def_readwrite("is_anomaly", &ml::AnomalyDetector::AnomalyResult::is_anomaly)
        .def_readwrite("anomaly_score", &ml::AnomalyDetector::AnomalyResult::anomaly_score)
        .def_readwrite("anomaly_features", &ml::AnomalyDetector::AnomalyResult::anomaly_features);
    
    py::class_<ml::AnomalyDetector, ml::MLModel, std::shared_ptr<ml::AnomalyDetector>>(ml_module, "AnomalyDetector")
        .def(py::init<>())
        .def("detect", &ml::AnomalyDetector::detect);
    
    // MLOptimizer
    py::enum_<ml::MLOptimizer::OptimizationObjective::Type>(ml_module, "OptimizationObjectiveType")
        .value("MINIMIZE_PRESSURE_DROP", ml::MLOptimizer::OptimizationObjective::MINIMIZE_PRESSURE_DROP)
        .value("MAXIMIZE_FLOW_RATE", ml::MLOptimizer::OptimizationObjective::MAXIMIZE_FLOW_RATE)
        .value("MINIMIZE_ENERGY_CONSUMPTION", ml::MLOptimizer::OptimizationObjective::MINIMIZE_ENERGY_CONSUMPTION)
        .value("CUSTOM", ml::MLOptimizer::OptimizationObjective::CUSTOM);
    
    py::class_<ml::MLOptimizer::OptimizationObjective>(ml_module, "OptimizationObjective")
        .def(py::init<>())
        .def_readwrite("type", &ml::MLOptimizer::OptimizationObjective::type)
        .def_readwrite("custom_function", &ml::MLOptimizer::OptimizationObjective::custom_function);
    
    py::class_<ml::MLOptimizer::OptimizationConstraints>(ml_module, "OptimizationConstraints")
        .def(py::init<>())
        .def_readwrite("min_pressure", &ml::MLOptimizer::OptimizationConstraints::min_pressure)
        .def_readwrite("max_pressure", &ml::MLOptimizer::OptimizationConstraints::max_pressure)
        .def_readwrite("max_velocity", &ml::MLOptimizer::OptimizationConstraints::max_velocity)
        .def_readwrite("node_flow_demands", &ml::MLOptimizer::OptimizationConstraints::node_flow_demands);
    
    py::class_<ml::MLOptimizer::OptimizationResult>(ml_module, "OptimizationResult")
        .def(py::init<>())
        .def_readwrite("success", &ml::MLOptimizer::OptimizationResult::success)
        .def_readwrite("objective_value", &ml::MLOptimizer::OptimizationResult::objective_value)
        .def_readwrite("pump_speeds", &ml::MLOptimizer::OptimizationResult::pump_speeds)
        .def_readwrite("valve_openings", &ml::MLOptimizer::OptimizationResult::valve_openings)
        .def_readwrite("compressor_ratios", &ml::MLOptimizer::OptimizationResult::compressor_ratios);
    
    py::class_<ml::MLOptimizer>(ml_module, "MLOptimizer")
        .def(py::init<>())
        .def("optimize", &ml::MLOptimizer::optimize);
    
    // DataDrivenCorrelation
    py::class_<ml::DataDrivenCorrelation, FlowCorrelation>(ml_module, "DataDrivenCorrelation")
        .def(py::init<>())
        .def("train", &ml::DataDrivenCorrelation::train)
        .def("save_model", &ml::DataDrivenCorrelation::save_model)
        .def("load_model", &ml::DataDrivenCorrelation::load_model);
    
    // DigitalTwin
    py::class_<ml::DigitalTwin::EstimatedState>(ml_module, "EstimatedState")
        .def_readwrite("node_pressures", &ml::DigitalTwin::EstimatedState::node_pressures)
        .def_readwrite("pipe_flows", &ml::DigitalTwin::EstimatedState::pipe_flows)
        .def_readwrite("uncertainties", &ml::DigitalTwin::EstimatedState::uncertainties);
    
    py::class_<ml::DigitalTwin::Discrepancy>(ml_module, "Discrepancy")
        .def_readwrite("location", &ml::DigitalTwin::Discrepancy::location)
        .def_readwrite("type", &ml::DigitalTwin::Discrepancy::type)
        .def_readwrite("severity", &ml::DigitalTwin::Discrepancy::severity)
        .def_readwrite("confidence", &ml::DigitalTwin::Discrepancy::confidence);
    
    py::class_<ml::DigitalTwin>(ml_module, "DigitalTwin")
        .def(py::init<>())
        .def("initialize", &ml::DigitalTwin::initialize)
        .def("update_with_measurements", &ml::DigitalTwin::update_with_measurements)
        .def("estimate_state", &ml::DigitalTwin::estimate_state)
        .def("predict_future", &ml::DigitalTwin::predict_future)
        .def("detect_discrepancies", &ml::DigitalTwin::detect_discrepancies);
    
    // ========== Module Functions ==========
    
    m.def("initialize", &initialize, "Initialize the library");
    m.def("get_version", &get_version, "Get version string");
    
    // Module attributes
    m.attr("__version__") = VERSION;
    m.attr("VERSION_MAJOR") = VERSION_MAJOR;
    m.attr("VERSION_MINOR") = VERSION_MINOR;
    m.attr("VERSION_PATCH") = VERSION_PATCH;
    
    // Constants
    auto constants = m.def_submodule("constants", "Physical constants");
    constants.attr("PI") = constants::PI;
    constants.attr("GRAVITY") = constants::GRAVITY;
    constants.attr("GAS_CONSTANT") = constants::GAS_CONSTANT;
    constants.attr("STANDARD_PRESSURE") = constants::STANDARD_PRESSURE;
    constants.attr("STANDARD_TEMPERATURE") = constants::STANDARD_TEMPERATURE;
    constants.attr("oil_water_tension") = constants::oil_water_tension;
}
