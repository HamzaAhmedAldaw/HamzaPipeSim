#include "pipeline_sim/node.h"

namespace pipeline_sim {

Node::Node(const std::string& id, NodeType type)
    : id_(id),
      type_(type),
      pressure_(101325.0),
      temperature_(288.15),
      elevation_(0.0),
      has_pressure_bc_(false),
      pressure_bc_(0.0),
      fixed_flow_rate_(0.0),
      pump_speed_(1.0),
      pump_curve_a_(0.0),
      pump_curve_b_(0.0),
      compressor_ratio_(1.0) {
}

} // namespace pipeline_sim