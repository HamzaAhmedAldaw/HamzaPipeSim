// AI_GENERATED: Network unit tests
#include <gtest/gtest.h>
#include "pipeline_sim/network.h"

using namespace pipeline_sim;

TEST(NetworkTest, AddNode) {
    Network network;
    auto node = network.add_node("test_node", NodeType::JUNCTION);
    
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->id(), "test_node");
    EXPECT_EQ(node->type(), NodeType::JUNCTION);
}

TEST(NetworkTest, AddPipe) {
    Network network;
    auto n1 = network.add_node("n1", NodeType::SOURCE);
    auto n2 = network.add_node("n2", NodeType::SINK);
    
    auto pipe = network.add_pipe("pipe1", n1, n2, 1000.0, 0.3);
    
    ASSERT_NE(pipe, nullptr);
    EXPECT_EQ(pipe->id(), "pipe1");
    EXPECT_EQ(pipe->length(), 1000.0);
    EXPECT_EQ(pipe->diameter(), 0.3);
}

TEST(NetworkTest, NodeConnectivity) {
    Network network;
    auto n1 = network.add_node("n1", NodeType::SOURCE);
    auto n2 = network.add_node("n2", NodeType::JUNCTION);
    auto n3 = network.add_node("n3", NodeType::SINK);
    
    network.add_pipe("p1", n1, n2, 500, 0.2);
    network.add_pipe("p2", n2, n3, 500, 0.2);
    
    auto upstream = network.get_upstream_pipes(n2);
    auto downstream = network.get_downstream_pipes(n2);
    
    EXPECT_EQ(upstream.size(), 1);
    EXPECT_EQ(downstream.size(), 1);
}