#pragma once

#include "slbjnv/common/types.hpp"
#include "slbjnv/model/batch_column.hpp"

#include <optional>
#include <string>

namespace slbjnv {

struct BranchVariable {
    BranchingRuleKind rule{BranchingRuleKind::item_selected};
    ItemId first_item{-1};
    ItemId second_item{-1};
    MachineId machine{-1};

    [[nodiscard]] std::string key() const;
};

struct PricingPairRestriction {
    ItemId first_item{};
    ItemId second_item{};
    BranchDirection direction{BranchDirection::less_equal_zero};
};

struct BranchConstraint {
    BranchVariable variable;
    BranchDirection direction{BranchDirection::less_equal_zero};

    [[nodiscard]] double coefficient(const BatchColumn& column) const;
    [[nodiscard]] std::optional<PricingPairRestriction> pricing_pair_restriction() const;
};

}  // namespace slbjnv
