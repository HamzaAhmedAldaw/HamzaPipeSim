#pragma once

#include "pipeline_sim/types.h"
#include "pipeline_sim/fluid_properties.h"
#include "pipeline_sim/pipe.h"
#include <string>
#include <map>

namespace pipeline_sim {

/// Flow pattern enumeration
enum class FlowPattern {
    SEGREGATED = 0,
    INTERMITTENT = 1,
    DISTRIBUTED = 2,
    ANNULAR = 3,
    BUBBLE = 4,
    SLUG = 5,
    CHURN = 6,
    MIST = 7,
    SINGLE_PHASE = 8
};

/// Base correlation interface
class FlowCorrelation {
public:
    struct Results {
        Real pressure_gradient;    // Pa/m
        Real liquid_holdup;       // fraction
        FlowPattern flow_pattern;
        Real friction_factor;
        Real mixture_density;     // kg/m³
        Real mixture_velocity;    // m/s
        std::map<std::string, Real> additional_data;
    };
    
    virtual ~FlowCorrelation() = default;
    
    /// Calculate pressure drop and flow characteristics
    virtual Results calculate(
        const FluidProperties& fluid,
        const Pipe& pipe,
        Real flow_rate,
        Real inlet_pressure,
        Real inlet_temperature
    ) const = 0;
    
    /// Get correlation name
    virtual std::string name() const = 0;
    
    /// Check if correlation is applicable
    virtual bool is_applicable(
        const FluidProperties& fluid,
        const Pipe& pipe
    ) const { return true; }
};

/// Single phase flow correlation
class SinglePhaseFlow : public FlowCorrelation {
public:
    Results calculate(
        const FluidProperties& fluid,
        const Pipe& pipe,
        Real flow_rate,
        Real inlet_pressure,
        Real inlet_temperature
    ) const override;
    
    std::string name() const override { return "Single Phase"; }
    
    bool is_applicable(
        const FluidProperties& fluid,
        const Pipe& pipe
    ) const override {
        return !fluid.is_multiphase();
    }
};

/// Core-annular flow correlation for heavy oil-water flow
class CoreAnnularFlow : public FlowCorrelation {
public:
    Results calculate(
        const FluidProperties& fluid,
        const Pipe& pipe,
        Real flow_rate,
        Real inlet_pressure,
        Real inlet_temperature
    ) const override;
    
    std::string name() const override { return "Core-Annular Flow"; }
    
    bool is_applicable(
        const FluidProperties& fluid,
        const Pipe& pipe
    ) const override {
        // For oil-water flow with high oil viscosity
        return fluid.has_oil && fluid.has_water && !fluid.has_gas &&
               fluid.oil_viscosity > 0.1;  // Pa.s
    }
};

/// Beggs-Brill correlation for multiphase flow
class BeggsBrill : public FlowCorrelation {
public:
    Results calculate(
        const FluidProperties& fluid,
        const Pipe& pipe,
        Real flow_rate,
        Real inlet_pressure,
        Real inlet_temperature
    ) const override;
    
    std::string name() const override { return "Beggs-Brill"; }
    
private:
    FlowPattern determine_flow_pattern(
        Real liquid_velocity,
        Real gas_velocity,
        Real pipe_diameter,
        Real inclination
    ) const;
    
    Real calculate_liquid_holdup(
        Real no_slip_holdup,
        Real froude_number,
        Real inclination,
        FlowPattern pattern
    ) const;
};

/// Beggs-Brill correlation (alias for compatibility)
using BeggsBrillCorrelation = BeggsBrill;

/// Hagedorn-Brown correlation for vertical wells
class HagedornBrownCorrelation : public FlowCorrelation {
public:
    Results calculate(
        const FluidProperties& fluid,
        const Pipe& pipe,
        Real flow_rate,
        Real inlet_pressure,
        Real inlet_temperature
    ) const override;
    
    std::string name() const override { return "Hagedorn-Brown"; }
    
    bool is_applicable(
        const FluidProperties& fluid,
        const Pipe& pipe
    ) const override {
        // Primarily for vertical or near-vertical flow
        return std::abs(pipe.inclination()) > 1.0; // > ~57 degrees
    }
    
private:
    Real griffith_wallis_correlation(
        Real liquid_velocity,
        Real gas_velocity,
        Real pipe_diameter
    ) const;
    
    Real cnu_correlation_number(
        Real liquid_viscosity,
        Real liquid_density,
        Real surface_tension,
        Real pipe_diameter
    ) const;
};

/// Gray correlation for wet gas
class GrayCorrelation : public FlowCorrelation {
public:
    Results calculate(
        const FluidProperties& fluid,
        const Pipe& pipe,
        Real flow_rate,
        Real inlet_pressure,
        Real inlet_temperature
    ) const override;
    
    std::string name() const override { return "Gray"; }
    
    bool is_applicable(
        const FluidProperties& fluid,
        const Pipe& pipe
    ) const override {
        // For high gas fraction flows
        return fluid.gas_fraction > 0.9;
    }
};

/// Mechanistic model (unified approach)
class MechanisticModel : public FlowCorrelation {
public:
    Results calculate(
        const FluidProperties& fluid,
        const Pipe& pipe,
        Real flow_rate,
        Real inlet_pressure,
        Real inlet_temperature
    ) const override;
    
    std::string name() const override { return "Mechanistic"; }
    
private:
    struct FlowRegime {
        FlowPattern pattern;
        Real holdup;
        Real interfacial_friction;
    };
    
    FlowRegime determine_flow_regime(
        const FluidProperties& fluid,
        const Pipe& pipe,
        Real liquid_velocity,
        Real gas_velocity
    ) const;
    
    Real stratified_holdup(
        Real liquid_flow,
        Real gas_flow,
        Real pipe_diameter,
        Real inclination
    ) const;
};

/// Correlation factory
class CorrelationFactory {
public:
    static std::unique_ptr<FlowCorrelation> create(const std::string& name);
    static std::vector<std::string> available_correlations();
    
    /// Register custom correlation
    static void register_correlation(
        const std::string& name,
        std::function<std::unique_ptr<FlowCorrelation>()> creator
    );
    
private:
    static std::map<std::string, std::function<std::unique_ptr<FlowCorrelation>()>>& registry();
};

} // namespace pipeline_sim