#include "branching_test_common.hpp"

int main() {
    using namespace slbjnv;
    branching_test::Fixture fixture;
    ItemMachineBranchingRule rule;
    const auto candidates = rule.candidates(fixture.solution, fixture.tolerances);
    assert(candidates.size() == 4);
    const BranchVariable variable{BranchingRuleKind::item_machine, 0, -1, 0};
    branching_test::find(candidates, variable);
    branching_test::check_children(variable, fixture.columns[0], fixture.columns[1]);
    return 0;
}
