#include "pipeline_sim/pipe.h"
#include <cmath>
#include <algorithm>

namespace pipeline_sim {

Pipe::Pipe(const std::string& id,
           Ptr<Node> upstream,
           Ptr<Node> downstream,
           Real length,
           Real diameter,
           Real roughness)
    : id_(id),
      upstream_(upstream),
      downstream_(downstream),
      length_(length),
      diameter_(diameter),
      roughness_(roughness),
      inclination_(0.0),
      flow_rate_(0.0),
      velocity_(0.0),
      wall_temperature_(288.15),
      heat_transfer_coefficient_(0.0),
      has_flow_bc_(false),
      flow_bc_(0.0),
      has_valve_(false),
      valve_opening_(1.0) {
}

Real Pipe::area() const {
    return constants::PI * diameter_ * diameter_ / 4.0;
}

Real Pipe::reynolds_number(Real density, Real viscosity) const {
    return density * std::abs(velocity_) * diameter_ / viscosity;
}

Real Pipe::volume() const {
    return area() * length_;
}

Real Pipe::friction_factor(Real reynolds) const {
    if (reynolds < 2300.0) {
        // Laminar flow
        return 64.0 / reynolds;
    } else {
        // Turbulent flow - Colebrook-White equation
        // Using Swamee-Jain explicit approximation
        Real A = -2.0 * std::log10(roughness_ / (3.7 * diameter_) + 5.74 / std::pow(reynolds, 0.9));
        return 0.25 / (A * A);
    }
}

} // namespace pipeline_sim
