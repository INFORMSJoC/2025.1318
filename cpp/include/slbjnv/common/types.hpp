#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace slbjnv {

using ItemId = std::int32_t;
using MachineId = std::int32_t;
using NodeId = std::int64_t;
using ColumnId = std::int64_t;

constexpr NodeId kNoNode = -1;
constexpr double kNegativeInfinity = -std::numeric_limits<double>::infinity();

enum class SolveStatus {
    not_solved,
    optimal,
    infeasible,
    unbounded,
    time_limit,
    numerical_error,
    backend_unavailable
};

enum class BranchDirection { less_equal_zero, greater_equal_one };
enum class BranchingRuleKind { item_selected = 1, master_selected = 2, item_machine = 3,
                               master_machine = 4, pair_together = 5 };
enum class SearchMode { baseline, collect_training_data, slbjnv };

struct NumericalTolerances {
    double reduced_cost{1.0e-7};
    double integrality{1.0e-6};
    double feasibility{1.0e-8};
    double bound{1.0e-8};
};

}  // namespace slbjnv
