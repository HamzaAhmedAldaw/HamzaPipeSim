/// AI_GENERATED: Common type definitions
/// Generated on: 2025-06-27
#pragma once

#include <Eigen/Core>
#include <complex>
#include <vector>
#include <memory>

namespace pipeline_sim {

// Floating point precision
using Real = double;
using Complex = std::complex<Real>;

// Matrix and vector types
using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
using SparseMatrix = Eigen::SparseMatrix<Real>;

// Smart pointers
template<typename T>
using Ptr = std::shared_ptr<T>;

template<typename T>
using UniquePtr = std::unique_ptr<T>;

// Common constants
namespace constants {
    constexpr Real PI = 3.14159265358979323846;
    constexpr Real GRAVITY = 9.80665;  // m/s²
    constexpr Real GAS_CONSTANT = 8.314;  // J/(mol·K)
    constexpr Real STANDARD_PRESSURE = 101325.0;  // Pa
    constexpr Real STANDARD_TEMPERATURE = 288.15;  // K
}

} // namespace pipeline_sim