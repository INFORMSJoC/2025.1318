#include "branching_test_common.hpp"

int main() {
    using namespace slbjnv;
    branching_test::Fixture fixture;
    MasterSelectedBranchingRule rule;
    const auto candidates = rule.candidates(fixture.solution, fixture.tolerances);
    assert(candidates.size() == 2);
    const BranchVariable variable{BranchingRuleKind::master_selected, 0, -1, -1};
    branching_test::find(candidates, variable);
    branching_test::check_children(variable, fixture.columns[0], fixture.columns[1]);
    return 0;
}
