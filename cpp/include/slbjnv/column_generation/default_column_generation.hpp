#pragma once

#include "slbjnv/column_generation/column_generation.hpp"
#include "slbjnv/instance/gbd_instance.hpp"

namespace slbjnv {

struct ColumnGenerationConfig {
    double reduced_cost_tolerance{1.0e-7};
    std::size_t maximum_iterations{10000};
};

class DefaultColumnGeneration final : public ColumnGeneration {
public:
    DefaultColumnGeneration(const GBDInstance& instance, ColumnGenerationConfig config = {})
        : instance_(instance), config_(config) {}

    [[nodiscard]] ColumnGenerationResult solve(
        RestrictedMasterProblem& master, PricingProblem& pricing,
        const std::vector<BranchConstraint>& inherited_constraints) override;

private:
    const GBDInstance& instance_;
    ColumnGenerationConfig config_;
};

}  // namespace slbjnv
