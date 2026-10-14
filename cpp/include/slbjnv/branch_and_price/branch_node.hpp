#pragma once

#include "slbjnv/branching/branching_rule.hpp"
#include "slbjnv/column_generation/column_generation.hpp"

#include <optional>
#include <vector>

namespace slbjnv {

struct BranchNode {
    NodeId id{kNoNode};
    NodeId parent_id{kNoNode};
    std::size_t depth{};
    std::vector<BranchConstraint> constraints;
    std::vector<BatchColumn> columns;
    ColumnGenerationResult relaxation;
    double upper_bound{kNegativeInfinity};
    double global_lower_bound{kNegativeInfinity};
    std::size_t exploration_index{};
    double elapsed_seconds{};
    bool integral{};
    bool active{};
    std::vector<NodeId> child_ids;
    std::optional<BranchVariable> branching_variable;
    std::size_t decision_index{};
};

}  // namespace slbjnv
