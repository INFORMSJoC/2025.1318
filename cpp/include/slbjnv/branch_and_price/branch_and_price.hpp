#pragma once

#include "slbjnv/branch_and_price/branching_policy.hpp"
#include "slbjnv/branch_and_price/incumbent_manager.hpp"
#include "slbjnv/branch_and_price/search_tree.hpp"

#include <array>

namespace slbjnv {

struct BoundSnapshot {
    std::size_t explored_nodes{};
    double elapsed_seconds{};
    double lower_bound{kNegativeInfinity};
    double upper_bound{kNegativeInfinity};
};

struct BranchAndPriceResult {
    SolveStatus status{SolveStatus::not_solved};
    double best_integer_objective{kNegativeInfinity};
    double global_upper_bound{kNegativeInfinity};
    std::size_t explored_nodes{};
    std::size_t generated_nodes{};
    std::size_t column_generation_iterations{};
    std::size_t generated_columns{};
    std::size_t branching_decisions{};
    std::array<std::size_t, 5> branching_rule_counts{};
    double inference_seconds{};
    std::size_t predicted_pairs{};
    double total_runtime_seconds{};
    std::vector<IncumbentSnapshot> incumbent_history;
    std::vector<BoundSnapshot> bound_history;
    SearchTree tree;
};

class BranchAndPrice {
public:
    virtual ~BranchAndPrice() = default;
    [[nodiscard]] virtual BranchAndPriceResult solve(const GBDInstance& instance, SearchMode mode) = 0;
};

}  // namespace slbjnv
