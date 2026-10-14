#pragma once

#include "slbjnv/column_generation/pricing_problem.hpp"
#include "slbjnv/model/cplex_c_api.hpp"

namespace slbjnv {

class CplexPricingProblem final : public PricingProblem {
public:
    explicit CplexPricingProblem(std::shared_ptr<CplexCApi> api) : api_(std::move(api)) {}

    [[nodiscard]] PricingResult solve(
        const GBDInstance& instance, MachineId machine, ItemId master,
        const MasterDuals& duals,
        const std::vector<BranchConstraint>& inherited_constraints) override;

private:
    std::shared_ptr<CplexCApi> api_;
};

}  // namespace slbjnv
