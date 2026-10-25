#include "slbjnv/column_generation/default_column_generation.hpp"

#include <utility>

namespace slbjnv {

ColumnGenerationResult DefaultColumnGeneration::solve(
    RestrictedMasterProblem& master, PricingProblem& pricing,
    const std::vector<BranchConstraint>& constraints) {
    ColumnGenerationResult result;
    for (std::size_t iteration = 0; iteration < config_.maximum_iterations; ++iteration) {
        result.master = master.solve_relaxation();
        result.iterations = iteration + 1;
        result.status = result.master.lp.status;
        if (result.status != SolveStatus::optimal) return result;
        if (iteration == 0) result.first_iteration_duals = result.master.duals;

        std::size_t added = 0;
        for (const auto& machine : instance_.machines) {
            for (const auto& item : instance_.items) {
                if (item.weight > machine.capacity + 1.0e-8) continue;
                auto priced = pricing.solve(instance_, machine.id, item.id, result.master.duals, constraints);
                if (iteration == 0 && priced.status == SolveStatus::optimal) {
                    result.first_iteration_reduced_costs.push_back(priced.reduced_cost);
                }
                if (priced.status == SolveStatus::infeasible) continue;
                if (priced.status != SolveStatus::optimal) {
                    result.status = priced.status;
                    return result;
                }
                if (priced.reduced_cost <= config_.reduced_cost_tolerance || !priced.column) continue;
                if (master.add_column(std::move(*priced.column))) {
                    ++added;
                    ++result.generated_columns;
                }
            }
        }
        if (added == 0) {
            result.converged = true;
            result.status = SolveStatus::optimal;
            return result;
        }
    }
    result.status = SolveStatus::time_limit;
    return result;
}

}  // namespace slbjnv
