#include "slbjnv/branch_and_price/incumbent_manager.hpp"
#include "slbjnv/branching/branch_constraint.hpp"
#include "slbjnv/features/feature_extractor.hpp"

#include <cassert>

int main() {
    using namespace slbjnv;
    static_assert(FeatureExtractor::kDimension == 1000);
    std::size_t total = 0;
    for (const auto& slice : FeatureExtractor::schema()) total += slice.length;
    assert(total == FeatureExtractor::kDimension);

    BatchColumn column{0, 1, 2, {2, 4}, 10.0};
    BranchConstraint rule5{{BranchingRuleKind::pair_together, 2, 4, -1},
                           BranchDirection::less_equal_zero};
    assert(rule5.coefficient(column) == 1.0);
    assert(rule5.pricing_pair_restriction().has_value());

    IncumbentManager incumbent;
    assert(incumbent.update(1, 10.0, 0.1));
    assert(!incumbent.update(2, 9.0, 0.2));
    assert(incumbent.update(3, 11.0, 0.3));
    assert(incumbent.lower_bound() == 11.0);
    return 0;
}
