#include "slbjnv/branching/branch_constraint.hpp"

#include <sstream>

namespace slbjnv {

std::string BranchVariable::key() const {
    std::ostringstream stream;
    stream << static_cast<int>(rule) << ':' << first_item << ':' << second_item << ':' << machine;
    return stream.str();
}

double BranchConstraint::coefficient(const BatchColumn& column) const {
    switch (variable.rule) {
        case BranchingRuleKind::item_selected:
            return column.contains(variable.first_item) ? 1.0 : 0.0;
        case BranchingRuleKind::master_selected:
            return column.master == variable.first_item ? 1.0 : 0.0;
        case BranchingRuleKind::item_machine:
            return column.machine == variable.machine && column.contains(variable.first_item) ? 1.0 : 0.0;
        case BranchingRuleKind::master_machine:
            return column.machine == variable.machine && column.master == variable.first_item ? 1.0 : 0.0;
        case BranchingRuleKind::pair_together:
            return column.contains(variable.first_item) && column.contains(variable.second_item) ? 1.0 : 0.0;
    }
    return 0.0;
}

std::optional<PricingPairRestriction> BranchConstraint::pricing_pair_restriction() const {
    if (variable.rule != BranchingRuleKind::pair_together) return std::nullopt;
    return PricingPairRestriction{variable.first_item, variable.second_item, direction};
}

}  // namespace slbjnv
