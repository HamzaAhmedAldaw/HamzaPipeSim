#pragma once

#include "pipeline_sim/types.h"
#include <string>

namespace pipeline_sim {

// Formatting functions
std::string format_pressure(Real pressure);
std::string format_temperature(Real temperature);
std::string format_flow_rate(Real flow_rate);

// Unit conversion functions
Real convert_bar_to_pa(Real bar);
Real convert_pa_to_bar(Real pa);
Real convert_celsius_to_kelvin(Real celsius);
Real convert_kelvin_to_celsius(Real kelvin);

} // namespace pipeline_sim