#pragma once

#include "slbjnv/branch_and_price/branch_node.hpp"

#include <optional>
#include <vector>

namespace slbjnv {

struct JointBranchDecision {
    NodeId node_id{kNoNode};
    BranchCandidate candidate;
    double score{};
};

class BranchingPolicy {
public:
    virtual ~BranchingPolicy() = default;
    [[nodiscard]] virtual std::optional<JointBranchDecision> select(
        const std::vector<const BranchNode*>& active_nodes) = 0;
};

}  // namespace slbjnv
