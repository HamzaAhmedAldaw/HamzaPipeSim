// ===== pipe.cpp =====
#include "pipeline_sim/pipe.h"
#include <cmath>

namespace pipeline_sim {

Pipe::Pipe(const std::string& id, 
           Ptr<Node> upstream, 
           Ptr<Node> downstream,
           Real length,
           Real diameter)
    : id_(id), 
      upstream_(upstream), 
      downstream_(downstream),
      length_(length), 
      diameter_(diameter) {
}

Real Pipe::reynolds_number(Real viscosity, Real density) const {
    if (viscosity <= 0) return 0;
    return density * std::abs(velocity()) * diameter_ / viscosity;
}

Real Pipe::friction_factor(Real reynolds) const {
    if (reynolds <= 0) return 0.064;  // Laminar flow default
    
    // Colebrook-White equation (simplified explicit approximation)
    if (reynolds < 2300) {
        // Laminar flow
        return 64.0 / reynolds;
    } else {
        // Turbulent flow - Swamee-Jain approximation
        Real relative_roughness = roughness_ / diameter_;
        Real a = -2.0 * std::log10(relative_roughness / 3.7 + 5.74 / std::pow(reynolds, 0.9));
        return 0.25 / (a * a);
    }
}

} // namespace pipeline_sim