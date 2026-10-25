#include "slbjnv/branch_and_price/default_branch_and_price.hpp"
#include "slbjnv/column_generation/cplex_pricing_problem.hpp"
#include "slbjnv/column_generation/cplex_restricted_master_problem.hpp"
#include "slbjnv/column_generation/default_column_generation.hpp"
#include "slbjnv/instance/instance_io.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <vector>

namespace {

std::vector<slbjnv::BatchColumn> enumerate_columns(const slbjnv::GBDInstance& instance) {
    std::vector<slbjnv::BatchColumn> result;
    const auto count = instance.items.size();
    assert(count < 63);
    for (const auto& machine : instance.machines) {
        for (const auto& master : instance.items) {
            for (std::uint64_t mask = 0; mask < (std::uint64_t{1} << count); ++mask) {
                if ((mask & (std::uint64_t{1} << master.id)) == 0) continue;
                slbjnv::BatchColumn column{-1, machine.id, master.id, {}, 0.0};
                bool compatible = true;
                double weight = 0.0;
                for (std::size_t i = 0; i < count; ++i) {
                    if ((mask & (std::uint64_t{1} << i)) == 0) continue;
                    if (!instance.is_compatible(master.id, static_cast<slbjnv::ItemId>(i))) { compatible = false; break; }
                    column.items.push_back(static_cast<slbjnv::ItemId>(i));
                    weight += instance.items[i].weight;
                    column.benefit += instance.items[i].profit;
                }
                if (compatible && weight <= machine.capacity + 1.0e-8) result.push_back(std::move(column));
            }
        }
    }
    return result;
}

double exhaustive_integer_optimum(const slbjnv::GBDInstance& instance,
                                  const std::vector<slbjnv::BatchColumn>& columns) {
    double best = 0.0;
    for (std::size_t first = 0; first < columns.size(); ++first) {
        best = std::max(best, columns[first].benefit);
        for (std::size_t second = first + 1; second < columns.size(); ++second) {
            if (columns[first].machine == columns[second].machine) continue;
            bool disjoint = true;
            for (const auto item : columns[first].items) if (columns[second].contains(item)) disjoint = false;
            if (disjoint) best = std::max(best, columns[first].benefit + columns[second].benefit);
        }
    }
    static_cast<void>(instance);
    return best;
}

}  // namespace

int main() {
    using namespace slbjnv;
    const auto path = std::filesystem::path(SLBJNV_SOURCE_DIR) / "tests/fixtures/gbd_2_6_debug.txt";
    const auto instance = GBDInstanceIO::read_file(path);
    const auto api = CplexCApi::load();
    assert(!api->version().empty());

    CplexRestrictedMasterProblem master(instance, {}, api);
    for (auto column : make_singleton_columns(instance)) assert(master.add_column(std::move(column)));
    CplexPricingProblem pricing(api);

    MasterDuals zero_duals;
    zero_duals.item_duals.assign(instance.items.size(), 0.0);
    zero_duals.machine_duals.assign(instance.machines.size(), 0.0);
    zero_duals.branch_duals = {0.0};

    const BranchVariable selected_item{BranchingRuleKind::item_selected, 1, -1, -1};
    const auto item_forbidden = pricing.solve(instance, 0, 0, zero_duals,
                                              {{selected_item, BranchDirection::less_equal_zero}});
    assert(item_forbidden.status == SolveStatus::optimal);
    assert(item_forbidden.column && !item_forbidden.column->contains(1));

    const BranchVariable selected_master{BranchingRuleKind::master_selected, 0, -1, -1};
    const auto master_forbidden = pricing.solve(instance, 0, 0, zero_duals,
                                                {{selected_master, BranchDirection::less_equal_zero}});
    assert(master_forbidden.status == SolveStatus::infeasible);
    const auto other_master = pricing.solve(instance, 0, 1, zero_duals,
                                            {{selected_master, BranchDirection::less_equal_zero}});
    assert(other_master.status == SolveStatus::optimal);

    const BranchVariable assigned_item{BranchingRuleKind::item_machine, 1, -1, 0};
    const auto assignment_forbidden = pricing.solve(instance, 0, 0, zero_duals,
                                                    {{assigned_item, BranchDirection::less_equal_zero}});
    assert(assignment_forbidden.status == SolveStatus::optimal);
    assert(assignment_forbidden.column && !assignment_forbidden.column->contains(1));
    const auto other_machine = pricing.solve(instance, 1, 0, zero_duals,
                                             {{assigned_item, BranchDirection::less_equal_zero}});
    assert(other_machine.status == SolveStatus::optimal);
    assert(other_machine.column && other_machine.column->contains(1));

    const BranchVariable assigned_master{BranchingRuleKind::master_machine, 0, -1, 0};
    const auto master_assignment_forbidden = pricing.solve(
        instance, 0, 0, zero_duals, {{assigned_master, BranchDirection::less_equal_zero}});
    assert(master_assignment_forbidden.status == SolveStatus::infeasible);
    const auto master_other_machine = pricing.solve(
        instance, 1, 0, zero_duals, {{assigned_master, BranchDirection::less_equal_zero}});
    assert(master_other_machine.status == SolveStatus::optimal);

    const BranchVariable pair{BranchingRuleKind::pair_together, 0, 1, -1};
    const auto apart = pricing.solve(instance, 0, 0, zero_duals,
                                     {{pair, BranchDirection::less_equal_zero}});
    assert(apart.status == SolveStatus::optimal);
    assert(apart.column && apart.column->contains(0) && !apart.column->contains(1));
    const auto together = pricing.solve(instance, 0, 0, zero_duals,
                                        {{pair, BranchDirection::greater_equal_one}});
    assert(together.status == SolveStatus::optimal);
    assert(together.column && together.column->contains(0) && together.column->contains(1));

    DefaultColumnGeneration cg(instance);
    const auto cg_result = cg.solve(master, pricing, {});
    assert(cg_result.status == SolveStatus::optimal);
    assert(cg_result.converged);
    assert(cg_result.generated_columns > 0);

    CplexRestrictedMasterProblem full_master(instance, {}, api);
    const auto all_columns = enumerate_columns(instance);
    for (auto column : all_columns) full_master.add_column(std::move(column));
    const auto full_lp = full_master.solve_relaxation();
    assert(full_lp.lp.status == SolveStatus::optimal);
    assert(std::abs(full_lp.lp.objective - cg_result.master.lp.objective) < 1.0e-6);

    DefaultBranchAndPrice solver(api);
    const auto bp = solver.solve(instance, SearchMode::baseline);
    assert(bp.status == SolveStatus::optimal);
    const auto integer_optimum = exhaustive_integer_optimum(instance, enumerate_columns(instance));
    assert(std::abs(bp.best_integer_objective - integer_optimum) < 1.0e-6);
    assert(std::abs(bp.global_upper_bound - integer_optimum) < 1.0e-6);
    assert(bp.total_runtime_seconds >= 0.0);
    assert(!bp.incumbent_history.empty());
    assert(!bp.bound_history.empty());

    const auto fractional_path = std::filesystem::path(SLBJNV_SOURCE_DIR) / "tests/fixtures/gbd_2_5_fractional.txt";
    const auto fractional_instance = GBDInstanceIO::read_file(fractional_path);
    const auto fractional_columns = enumerate_columns(fractional_instance);
    DefaultBranchAndPrice branching_solver(api);
    const auto branched = branching_solver.solve(fractional_instance, SearchMode::baseline);
    assert(branched.status == SolveStatus::optimal);
    assert(branched.branching_decisions > 0);
    assert(branched.explored_nodes > 1);
    const auto fractional_integer_optimum = exhaustive_integer_optimum(fractional_instance, fractional_columns);
    assert(std::abs(fractional_integer_optimum - 55.0) < 1.0e-6);
    assert(std::abs(branched.best_integer_objective - fractional_integer_optimum) < 1.0e-6);
    assert(branched.bound_history.size() == branched.explored_nodes);
    return 0;
}
