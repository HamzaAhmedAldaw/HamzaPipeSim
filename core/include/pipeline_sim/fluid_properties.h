#pragma once

#include "pipeline_sim/types.h"
#include <cmath>

namespace pipeline_sim {

/// Fluid properties structure
struct FluidProperties {
    // Phase densities (kg/m³)
    Real oil_density{850.0};
    Real gas_density{1.2};
    Real water_density{1000.0};
    
    // Phase viscosities (Pa.s)
    Real oil_viscosity{0.001};
    Real gas_viscosity{1.8e-5};
    Real water_viscosity{0.001};
    
    // Phase fractions (volumetric)
    Real oil_fraction{1.0};
    Real gas_fraction{0.0};
    Real water_fraction{0.0};
    
    // Surface tension (N/m)
    Real oil_water_tension{0.03};
    Real oil_gas_tension{0.02};
    
    // Temperature and pressure
    Real temperature{288.15};  // K
    Real pressure{101325.0};   // Pa
    
    // PVT properties
    Real gas_oil_ratio{0.0};  // sm³/sm³
    Real water_cut{0.0};      // fraction
    Real bubble_point_pressure{101325.0};  // Pa
    Real oil_formation_volume_factor{1.0};
    Real gas_formation_volume_factor{1.0};
    Real water_formation_volume_factor{1.0};
    
    // Phase presence flags
    bool has_oil{true};
    bool has_gas{false};
    bool has_water{false};
    
    /// Check if multiphase
    bool is_multiphase() const {
        int phase_count = 0;
        if (has_oil && oil_fraction > 0) phase_count++;
        if (has_gas && gas_fraction > 0) phase_count++;
        if (has_water && water_fraction > 0) phase_count++;
        return phase_count > 1;
    }
    
    /// Calculate mixture density
    Real mixture_density() const {
        return oil_density * oil_fraction +
               gas_density * gas_fraction +
               water_density * water_fraction;
    }
    
    /// Calculate mixture viscosity (simple volume-weighted)
    Real mixture_viscosity() const {
        return oil_viscosity * oil_fraction +
               gas_viscosity * gas_fraction +
               water_viscosity * water_fraction;
    }
    
    /// Calculate liquid fraction
    Real liquid_fraction() const {
        return oil_fraction + water_fraction;
    }
    
    /// Update formation volume factors based on pressure
    void update_pvt(Real pressure);
    
    /// Calculate Z-factor for gas
    Real gas_z_factor(Real pressure, Real temperature) const;
    
    /// Calculate oil viscosity at pressure
    Real oil_viscosity_at_pressure(Real pressure) const;
};

/// Black oil model for PVT calculations
class BlackOilModel {
public:
    /// Calculate oil formation volume factor
    static Real oil_fvf(Real pressure, Real bubble_pressure, 
                       Real gas_oil_ratio, Real temperature);
    
    /// Calculate gas formation volume factor
    static Real gas_fvf(Real pressure, Real temperature, Real z_factor);
    
    /// Calculate solution gas-oil ratio
    static Real solution_gor(Real pressure, Real temperature);
    
    /// Calculate oil viscosity
    static Real oil_viscosity(Real pressure, Real bubble_pressure,
                             Real dead_oil_viscosity, Real gas_oil_ratio);
};

/// Gas properties calculations
class GasProperties {
public:
    /// Calculate pseudo-critical properties for natural gas
    static void pseudo_critical_properties(Real specific_gravity,
                                         Real& pseudo_critical_pressure,
                                         Real& pseudo_critical_temperature);
    
    /// Calculate Z-factor using Dranchuk-Purvis-Robinson correlation
    static Real z_factor_dpr(Real pseudo_reduced_pressure,
                            Real pseudo_reduced_temperature);
    
    /// Calculate gas viscosity using Lee-Gonzalez-Eakin correlation
    static Real viscosity_lge(Real temperature, Real density,
                             Real molecular_weight);
};

/// Water properties calculations
class WaterProperties {
public:
    /// Calculate water formation volume factor
    static Real water_fvf(Real pressure, Real temperature);
    
    /// Calculate water viscosity
    static Real water_viscosity(Real temperature, Real salinity);
    
    /// Calculate water density
    static Real water_density(Real pressure, Real temperature, Real salinity);
};

} // namespace pipeline_sim