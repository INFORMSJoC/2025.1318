#pragma once

#include "slbjnv/branching/branching_rule.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

namespace branching_test {

struct Fixture {
    std::vector<slbjnv::BatchColumn> columns{
        {0, 0, 0, {0, 1}, 9.0},
        {1, 1, 1, {1, 2}, 11.0},
    };
    std::vector<slbjnv::FractionalColumnValue> solution{
        {&columns[0], 0.5}, {&columns[1], 0.5}
    };
    slbjnv::NumericalTolerances tolerances{};
};

inline const slbjnv::BranchCandidate& find(const std::vector<slbjnv::BranchCandidate>& candidates,
                                           const slbjnv::BranchVariable& variable) {
    const auto it = std::find_if(candidates.begin(), candidates.end(), [&](const auto& candidate) {
        return candidate.variable.key() == variable.key();
    });
    assert(it != candidates.end());
    assert(std::abs(it->lp_value - 0.5) < 1.0e-12);
    return *it;
}

inline void check_children(const slbjnv::BranchVariable& variable,
                           const slbjnv::BatchColumn& coefficient_one,
                           const slbjnv::BatchColumn& coefficient_zero) {
    const slbjnv::BranchConstraint left{variable, slbjnv::BranchDirection::less_equal_zero};
    const slbjnv::BranchConstraint right{variable, slbjnv::BranchDirection::greater_equal_one};
    assert(left.direction != right.direction);
    assert(left.coefficient(coefficient_one) == 1.0);
    assert(right.coefficient(coefficient_one) == 1.0);
    assert(left.coefficient(coefficient_zero) == 0.0);
    assert(right.coefficient(coefficient_zero) == 0.0);
}

}  // namespace branching_test
