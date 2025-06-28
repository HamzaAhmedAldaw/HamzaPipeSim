#include "pipeline_sim/node.h"

namespace pipeline_sim {

Node::Node(const std::string& id, NodeType type)
    : id_(id)
    , type_(type)
    , pressure_(101325.0)      // Standard atmospheric pressure in Pa
    , temperature_(288.15)     // Standard temperature in K (15°C)
    , elevation_(0.0)          // Sea level
    , has_pressure_bc_(false)
    , pressure_bc_(0.0)
    , fixed_flow_rate_(0.0)
    , pump_speed_(1.0)
    , pump_curve_a_(0.0)
    , pump_curve_b_(0.0)
    , compressor_ratio_(1.0) {
    // Constructor body - all member initialization done in initializer list
}

} // namespace pipeline_sim
