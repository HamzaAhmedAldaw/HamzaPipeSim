// ===== include/pipeline_sim/equipment.h =====
#pragma once

#include "pipeline_sim/types.h"
#include "pipeline_sim/node.h"
#include <functional>

namespace pipeline_sim {

/// Base equipment class
class Equipment {
public:
    Equipment(const std::string& id, NodeType type)
        : id_(id), type_(type) {}
    
    virtual ~Equipment() = default;
    
    /// Calculate outlet conditions given inlet
    virtual void calculate(
        Real inlet_pressure,
        Real inlet_temperature,
        Real flow_rate,
        Real& outlet_pressure,
        Real& outlet_temperature
    ) = 0;
    
    /// Get power consumption (W)
    virtual Real power_consumption() const { return 0.0; }
    
    /// Get efficiency
    virtual Real efficiency() const { return 1.0; }
    
    const std::string& id() const { return id_; }
    NodeType type() const { return type_; }
    
protected:
    std::string id_;
    NodeType type_;
};

/// Centrifugal pump model
class CentrifugalPump : public Equipment {
public:
    CentrifugalPump(const std::string& id);
    
    /// Set pump curve coefficients (H = a - b*Q - c*Q²)
    void set_curve_coefficients(Real a, Real b, Real c) {
        a_ = a; b_ = b; c_ = c;
    }
    
    /// Set speed ratio (actual/rated)
    void set_speed_ratio(Real ratio) { speed_ratio_ = ratio; }
    
    /// Set efficiency curve
    void set_efficiency_curve(
        std::function<Real(Real)> efficiency_func
    ) {
        efficiency_func_ = efficiency_func;
    }
    
    void calculate(
        Real inlet_pressure,
        Real inlet_temperature,
        Real flow_rate,
        Real& outlet_pressure,
        Real& outlet_temperature
    ) override;
    
    Real power_consumption() const override { return power_; }
    Real efficiency() const override { return efficiency_; }
    
    /// Get pump head at given flow
    Real head(Real flow_rate) const;
    
private:
    // Pump curve: H = a - b*Q - c*Q²
    Real a_{100.0};  // m
    Real b_{0.0};
    Real c_{1000.0};
    
    Real speed_ratio_{1.0};
    Real efficiency_{0.75};
    Real power_{0.0};
    
    std::function<Real(Real)> efficiency_func_;
};

/// Compressor model
class Compressor : public Equipment {
public:
    enum class Type {
        CENTRIFUGAL,
        RECIPROCATING,
        SCREW
    };
    
    Compressor(const std::string& id, Type type = Type::CENTRIFUGAL);
    
    /// Set performance map
    void set_performance_map(
        std::function<Real(Real, Real)> pressure_ratio_func,
        std::function<Real(Real, Real)> efficiency_func
    ) {
        pressure_ratio_func_ = pressure_ratio_func;
        efficiency_func_ = efficiency_func;
    }
    
    /// Set polytropic efficiency
    void set_polytropic_efficiency(Real eff) {
        polytropic_efficiency_ = eff;
    }
    
    void calculate(
        Real inlet_pressure,
        Real inlet_temperature,
        Real flow_rate,
        Real& outlet_pressure,
        Real& outlet_temperature
    ) override;
    
    Real power_consumption() const override { return power_; }
    Real efficiency() const override { return polytropic_efficiency_; }
    
private:
    Type type_;
    Real polytropic_efficiency_{0.75};
    Real power_{0.0};
    
    std::function<Real(Real, Real)> pressure_ratio_func_;
    std::function<Real(Real, Real)> efficiency_func_;
    
    /// Calculate compression work
    Real compression_work(
        Real inlet_pressure,
        Real outlet_pressure,
        Real inlet_temperature,
        Real gas_properties
    ) const;
};

/// Control valve model
class ControlValve : public Equipment {
public:
    ControlValve(const std::string& id);
    
    /// Set valve coefficient (Cv)
    void set_cv(Real cv) { cv_ = cv; }
    
    /// Set valve opening (0-1)
    void set_opening(Real opening) {
        opening_ = std::max(0.0, std::min(1.0, opening));
    }
    
    /// Set valve characteristic
    enum class Characteristic {
        LINEAR,
        EQUAL_PERCENTAGE,
        QUICK_OPENING
    };
    
    void set_characteristic(Characteristic char_type) {
        characteristic_ = char_type;
    }
    
    void calculate(
        Real inlet_pressure,
        Real inlet_temperature,
        Real flow_rate,
        Real& outlet_pressure,
        Real& outlet_temperature
    ) override;
    
    /// Calculate required Cv for given conditions
    static Real required_cv(
        Real flow_rate,
        Real pressure_drop,
        Real specific_gravity
    );
    
private:
    Real cv_{100.0};
    Real opening_{1.0};
    Characteristic characteristic_{Characteristic::LINEAR};
    
    /// Get effective Cv based on opening
    Real effective_cv() const;
};

/// Separator model
class Separator : public Equipment {
public:
    enum class Type {
        TWO_PHASE,    // Gas-liquid
        THREE_PHASE   // Gas-oil-water
    };
    
    Separator(const std::string& id, Type type = Type::TWO_PHASE);
    
    /// Set separator efficiency
    void set_separation_efficiency(Real gas_eff, Real liquid_eff) {
        gas_efficiency_ = gas_eff;
        liquid_efficiency_ = liquid_eff;
    }
    
    /// Set residence time
    void set_residence_time(Real time) { residence_time_ = time; }
    
    void calculate(
        Real inlet_pressure,
        Real inlet_temperature,
        Real flow_rate,
        Real& outlet_pressure,
        Real& outlet_temperature
    ) override;
    
    /// Get separated phase flow rates
    struct SeparatedFlows {
        Real gas_flow;
        Real oil_flow;
        Real water_flow;
    };
    
    SeparatedFlows get_separated_flows() const { return separated_flows_; }
    
private:
    Type type_;
    Real gas_efficiency_{0.99};
    Real liquid_efficiency_{0.95};
    Real residence_time_{120.0};  // seconds
    SeparatedFlows separated_flows_;
    
    /// Calculate separation based on residence time
    void calculate_separation(
        const FluidProperties& fluid,
        Real total_flow
    );
};

/// Heat exchanger model
class HeatExchanger : public Equipment {
public:
    enum class Type {
        SHELL_AND_TUBE,
        PLATE,
        AIR_COOLED
    };
    
    HeatExchanger(const std::string& id, Type type = Type::SHELL_AND_TUBE);
    
    /// Set heat transfer parameters
    void set_ua_value(Real ua) { ua_ = ua; }  // W/K
    
    /// Set cooling medium temperature
    void set_cooling_temperature(Real temp) { cooling_temp_ = temp; }
    
    void calculate(
        Real inlet_pressure,
        Real inlet_temperature,
        Real flow_rate,
        Real& outlet_pressure,
        Real& outlet_temperature
    ) override;
    
    /// Get heat duty
    Real heat_duty() const { return heat_duty_; }
    
private:
    Type type_;
    Real ua_{10000.0};  // W/K
    Real cooling_temp_{298.15};  // K
    Real heat_duty_{0.0};
    
    /// Calculate heat transfer
    Real calculate_heat_transfer(
        Real inlet_temp,
        Real flow_rate,
        Real specific_heat
    ) const;
};

} // namespace pipeline_sim