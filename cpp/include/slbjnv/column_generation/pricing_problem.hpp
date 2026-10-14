#pragma once

#include "slbjnv/column_generation/restricted_master_problem.hpp"
#include "slbjnv/instance/gbd_instance.hpp"

#include <optional>
#include <vector>

namespace slbjnv {

struct PricingResult {
    SolveStatus status{SolveStatus::not_solved};
    double reduced_cost{kNegativeInfinity};
    std::optional<BatchColumn> column;
};

class PricingProblem {
public:
    virtual ~PricingProblem() = default;
    [[nodiscard]] virtual PricingResult solve(
        const GBDInstance& instance,
        MachineId machine,
        ItemId master,
        const MasterDuals& duals,
        const std::vector<BranchConstraint>& inherited_constraints) = 0;
};

}  // namespace slbjnv
