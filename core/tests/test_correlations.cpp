
// AI_GENERATED: C++ correlation tests
#include <gtest/gtest.h>
#include "pipeline_sim/correlations.h"
#include <cmath>

using namespace pipeline_sim;

class CorrelationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test fluid
        fluid_.oil_density = 850.0;
        fluid_.gas_density = 0.85;
        fluid_.water_density = 1025.0;
        fluid_.oil_viscosity = 0.01;
        fluid_.gas_viscosity = 1.8e-5;
        fluid_.water_viscosity = 0.001;
        
        // Create test pipe
        auto n1 = std::make_shared<Node>("n1", NodeType::SOURCE);
        auto n2 = std::make_shared<Node>("n2", NodeType::SINK);
        pipe_ = std::make_shared<Pipe>("test_pipe", n1, n2, 1000.0, 0.2);
    }
    
    FluidProperties fluid_;
    Ptr<Pipe> pipe_;
};

TEST_F(CorrelationTest, BeggsBrillSinglePhase) {
    // Single phase oil
    fluid_.oil_fraction = 1.0;
    fluid_.gas_fraction = 0.0;
    fluid_.water_fraction = 0.0;
    
    BeggsBrillCorrelation correlation;
    auto result = correlation.calculate(fluid_, *pipe_, 0.05, 50e5, 300.0);
    
    EXPECT_NEAR(result.liquid_holdup, 1.0, 0.01);
    EXPECT_GT(result.pressure_gradient, 0.0);
    EXPECT_EQ(result.flow_pattern, FlowPattern::SEGREGATED);
}

TEST_F(CorrelationTest, BeggsBrillTwoPhase) {
    // Two-phase flow
    fluid_.oil_fraction = 0.7;
    fluid_.gas_fraction = 0.3;
    fluid_.water_fraction = 0.0;
    
    BeggsBrillCorrelation correlation;
    auto result = correlation.calculate(fluid_, *pipe_, 0.05, 50e5, 300.0);
    
    EXPECT_GE(result.liquid_holdup, 0.7);
    EXPECT_LE(result.liquid_holdup, 1.0);
    EXPECT_GT(result.pressure_gradient, 0.0);
}

TEST_F(CorrelationTest, CorrelationFactory) {
    auto beggs_brill = CorrelationFactory::create("Beggs-Brill");
    ASSERT_NE(beggs_brill, nullptr);
    EXPECT_EQ(beggs_brill->name(), "Beggs-Brill");
    
    auto available = CorrelationFactory::available_correlations();
    EXPECT_GE(available.size(), 4);
    
    // Test unknown correlation
    EXPECT_THROW(CorrelationFactory::create("Unknown"), std::runtime_error);
}

TEST_F(CorrelationTest, FlowPatternTransitions) {
    BeggsBrillCorrelation correlation;
    
    // Test flow pattern transitions with varying gas fraction
    std::vector<Real> gas_fractions = {0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95};
    FlowPattern last_pattern = FlowPattern::SEGREGATED;
    
    for (Real gf : gas_fractions) {
        fluid_.gas_fraction = gf;
        fluid_.oil_fraction = (1.0 - gf) * 0.8;
        fluid_.water_fraction = (1.0 - gf) * 0.2;
        
        auto result = correlation.calculate(fluid_, *pipe_, 0.1, 50e5, 300.0);
        
        // Flow pattern should change with gas fraction
        if (gf > 0.5) {
            EXPECT_NE(result.flow_pattern, FlowPattern::SEGREGATED);
        }
        
        last_pattern = result.flow_pattern;
    }
}