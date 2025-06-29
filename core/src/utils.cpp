#include "pipeline_sim/utils.h"
#include <sstream>
#include <iomanip>

namespace pipeline_sim {

std::string format_pressure(Real pressure) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << pressure / 1e5 << " bar";
    return oss.str();
}

std::string format_temperature(Real temperature) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << convert_kelvin_to_celsius(temperature) << " °C";
    return oss.str();
}

std::string format_flow_rate(Real flow_rate) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << flow_rate << " m³/s";
    return oss.str();
}

Real convert_bar_to_pa(Real bar) {
    return bar * 1e5;
}

Real convert_pa_to_bar(Real pa) {
    return pa / 1e5;
}

Real convert_celsius_to_kelvin(Real celsius) {
    return celsius + 273.15;
}

Real convert_kelvin_to_celsius(Real kelvin) {
    return kelvin - 273.15;
}

} // namespace pipeline_sim
