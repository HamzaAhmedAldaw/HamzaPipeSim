#include "pipeline_sim/fluid_properties.h"
#include <cmath>
#include <algorithm>

namespace pipeline_sim {

// FluidProperties methods
void FluidProperties::update_pvt(Real pressure) {
    // Simple PVT update
    if (pressure < bubble_point_pressure && has_gas) {
        // Below bubble point, gas comes out of solution
        Real pressure_ratio = pressure / bubble_point_pressure;
        gas_oil_ratio = gas_oil_ratio * pressure_ratio;
    }
    
    // Update formation volume factors
    oil_formation_volume_factor = BlackOilModel::oil_fvf(
        pressure, bubble_point_pressure, gas_oil_ratio, temperature
    );
    
    if (has_gas) {
        Real z = gas_z_factor(pressure, temperature);
        gas_formation_volume_factor = BlackOilModel::gas_fvf(pressure, temperature, z);
    }
    
    if (has_water) {
        water_formation_volume_factor = WaterProperties::water_fvf(pressure, temperature);
    }
}

Real FluidProperties::gas_z_factor(Real pressure, Real temperature) const {
    // Simplified Z-factor calculation
    Real pc, tc;
    GasProperties::pseudo_critical_properties(gas_density / 1.2, pc, tc);
    
    Real pr = pressure / pc;
    Real tr = temperature / tc;
    
    return GasProperties::z_factor_dpr(pr, tr);
}

Real FluidProperties::oil_viscosity_at_pressure(Real pressure) const {
    return BlackOilModel::oil_viscosity(
        pressure, bubble_point_pressure, oil_viscosity, gas_oil_ratio
    );
}

// BlackOilModel implementation
Real BlackOilModel::oil_fvf(Real pressure, Real bubble_pressure, 
                            Real gas_oil_ratio, Real temperature) {
    // Simplified Standing correlation
    if (pressure >= bubble_pressure) {
        // Undersaturated oil
        Real co = 1e-5;  // Oil compressibility, 1/Pa
        return 1.0 + co * (pressure - bubble_pressure);
    } else {
        // Saturated oil
        Real bob = 1.0 + 0.0001 * gas_oil_ratio;  // Simplified
        return bob * (pressure / bubble_pressure);
    }
}

Real BlackOilModel::gas_fvf(Real pressure, Real temperature, Real z_factor) {
    // Bg = ZT/P * Psc/Tsc
    Real psc = 101325.0;  // Standard pressure, Pa
    Real tsc = 288.15;    // Standard temperature, K
    
    return z_factor * temperature * psc / (pressure * tsc);
}

Real BlackOilModel::solution_gor(Real pressure, Real temperature) {
    // Simplified Standing correlation
    if (pressure < 101325.0) return 0.0;
    
    Real api = 30.0;  // Assumed API gravity
    Real yg = 0.65;   // Gas specific gravity
    
    Real x = 0.0125 * api - 0.00091 * (temperature - 273.15);
    Real rs = yg * std::pow(pressure / 1.8e5 * std::pow(10, x), 1.2048);
    
    return rs;
}

Real BlackOilModel::oil_viscosity(Real pressure, Real bubble_pressure,
                                 Real dead_oil_viscosity, Real gas_oil_ratio) {
    // Simplified Beggs-Robinson correlation
    if (pressure >= bubble_pressure) {
        // Undersaturated oil viscosity
        Real m = 2.6 * std::pow(pressure / 1e6, 1.187) * 
                std::exp(-11.513 - 8.98e-5 * pressure);
        return dead_oil_viscosity * std::pow(pressure / bubble_pressure, m);
    } else {
        // Saturated oil viscosity
        Real a = 10.715 * std::pow(gas_oil_ratio + 100, -0.515);
        Real b = 5.44 * std::pow(gas_oil_ratio + 150, -0.338);
        return a * std::pow(dead_oil_viscosity, b);
    }
}

// GasProperties implementation
void GasProperties::pseudo_critical_properties(Real specific_gravity,
                                             Real& pseudo_critical_pressure,
                                             Real& pseudo_critical_temperature) {
    // Standing's correlation
    if (specific_gravity < 0.75) {
        pseudo_critical_temperature = 168 + 325 * specific_gravity - 
                                     12.5 * specific_gravity * specific_gravity;
        pseudo_critical_pressure = 677 + 15.0 * specific_gravity - 
                                  37.5 * specific_gravity * specific_gravity;
    } else {
        pseudo_critical_temperature = 187 + 330 * specific_gravity - 
                                     71.5 * specific_gravity * specific_gravity;
        pseudo_critical_pressure = 706 - 51.7 * specific_gravity - 
                                  11.1 * specific_gravity * specific_gravity;
    }
    
    // Convert to SI units
    pseudo_critical_temperature = (pseudo_critical_temperature + 459.67) * 5.0/9.0;  // R to K
    pseudo_critical_pressure = pseudo_critical_pressure * 6894.76;  // psia to Pa
}

Real GasProperties::z_factor_dpr(Real pseudo_reduced_pressure,
                               Real pseudo_reduced_temperature) {
    // Dranchuk-Purvis-Robinson correlation (simplified)
    Real a1 = 0.3265, a2 = -1.0700, a3 = -0.5339;
    Real a4 = 0.01569, a5 = -0.05165, a6 = 0.5475;
    Real a7 = -0.7361, a8 = 0.1844, a9 = 0.1056;
    Real a10 = 0.6134, a11 = 0.7210;
    
    Real tr = pseudo_reduced_temperature;
    Real pr = pseudo_reduced_pressure;
    
    // Initial guess
    Real z = 1.0;
    
    // Newton-Raphson iteration (simplified - just one iteration)
    Real rho_r = 0.27 * pr / (z * tr);
    Real z_new = 1 + (a1 + a2/tr + a3/(tr*tr*tr)) * rho_r +
                (a4 + a5/tr) * rho_r * rho_r +
                a5 * a6 * rho_r * rho_r * rho_r * rho_r * rho_r / tr;
    
    return z_new;
}

Real GasProperties::viscosity_lge(Real temperature, Real density,
                                 Real molecular_weight) {
    // Lee-Gonzalez-Eakin correlation
    Real k = (9.4 + 0.02 * molecular_weight) * std::pow(temperature, 1.5) /
            (209 + 19 * molecular_weight + temperature);
    Real x = 3.5 + 986.0/temperature + 0.01 * molecular_weight;
    Real y = 2.4 - 0.2 * x;
    
    return k * std::exp(x * std::pow(density/1000.0, y)) * 1e-7;  // Convert to Pa.s
}

// WaterProperties implementation
Real WaterProperties::water_fvf(Real pressure, Real temperature) {
    // Simplified correlation
    Real bw = 1.0 + 1e-5 * (temperature - 288.15) - 1e-6 * (pressure - 101325);
    return std::max(0.98, std::min(1.02, bw));
}

Real WaterProperties::water_viscosity(Real temperature, Real salinity) {
    // Simplified correlation for water viscosity
    Real t_c = temperature - 273.15;  // Convert to Celsius
    Real mu_w = 1.002e-3 * std::exp(-0.02 * (t_c - 20));  // Fresh water
    
    // Salinity correction
    Real salinity_factor = 1.0 + 0.001 * salinity;
    
    return mu_w * salinity_factor;
}

Real WaterProperties::water_density(Real pressure, Real temperature, Real salinity) {
    // Simplified correlation
    Real rho_w = 1000.0;  // kg/m³ at standard conditions
    
    // Temperature correction
    Real t_c = temperature - 273.15;
    rho_w *= (1.0 - 0.0002 * (t_c - 4.0));
    
    // Pressure correction (water is nearly incompressible)
    rho_w *= (1.0 + 4.5e-10 * (pressure - 101325));
    
    // Salinity correction
    rho_w *= (1.0 + 0.0007 * salinity);
    
    return rho_w;
}

} // namespace pipeline_sim
