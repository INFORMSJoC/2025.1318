#pragma once

#include "slbjnv/features/feature_extractor.hpp"

#include <string>
#include <vector>

namespace slbjnv {

struct TrainingRecord {
    std::string instance_id;
    std::size_t decision_index{};
    NodeId node_id{kNoNode};
    NodeId parent_node_id{kNoNode};
    BranchVariable branching_variable;
    FeatureExtractor::FeatureVector feature_vector{};
    double target_score{};
    std::vector<NodeId> key_node_sequence;
    std::vector<double> key_value_sequence;
    double root_objective{};
};

}  // namespace slbjnv
