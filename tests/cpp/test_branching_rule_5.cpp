#include "branching_test_common.hpp"

int main() {
    using namespace slbjnv;
    branching_test::Fixture fixture;
    PairTogetherBranchingRule rule;
    const auto candidates = rule.candidates(fixture.solution, fixture.tolerances);
    assert(candidates.size() == 2);
    const BranchVariable variable{BranchingRuleKind::pair_together, 0, 1, -1};
    branching_test::find(candidates, variable);
    branching_test::check_children(variable, fixture.columns[0], fixture.columns[1]);
    const BranchConstraint left{variable, BranchDirection::less_equal_zero};
    const BranchConstraint right{variable, BranchDirection::greater_equal_one};
    assert(left.pricing_pair_restriction()->direction == BranchDirection::less_equal_zero);
    assert(right.pricing_pair_restriction()->direction == BranchDirection::greater_equal_one);
    return 0;
}
