#pragma once

#include "slbjnv/column_generation/pricing_problem.hpp"

#include <cstddef>

namespace slbjnv {

struct ColumnGenerationResult {
    SolveStatus status{SolveStatus::not_solved};
    MasterSolution master;
    std::size_t iterations{};
    std::size_t generated_columns{};
    bool converged{};
    MasterDuals first_iteration_duals;
    std::vector<double> first_iteration_reduced_costs;
};

class ColumnGeneration {
public:
    virtual ~ColumnGeneration() = default;
    [[nodiscard]] virtual ColumnGenerationResult solve(
        RestrictedMasterProblem& master,
        PricingProblem& pricing,
        const std::vector<BranchConstraint>& inherited_constraints) = 0;
};

}  // namespace slbjnv
