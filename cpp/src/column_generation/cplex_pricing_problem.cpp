#include "slbjnv/column_generation/cplex_pricing_problem.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace slbjnv {
namespace {

class ProblemGuard {
public:
    ProblemGuard(const CplexCApi& api, const char* name) : api_(api), problem_(api.create_problem(name)) {}
    ~ProblemGuard() { api_.free_problem(problem_); }
    [[nodiscard]] CplexProblem get() const { return problem_; }
private:
    const CplexCApi& api_;
    CplexProblem problem_{};
};

bool fixed_master_allows(const BranchConstraint& constraint, MachineId machine, ItemId master) {
    bool coefficient = false;
    switch (constraint.variable.rule) {
        case BranchingRuleKind::master_selected:
            coefficient = master == constraint.variable.first_item;
            break;
        case BranchingRuleKind::master_machine:
            coefficient = machine == constraint.variable.machine && master == constraint.variable.first_item;
            break;
        default:
            return true;
    }
    // A zero branch forbids every column with coefficient one. A one branch is
    // an aggregate RMP requirement; columns with coefficient zero remain valid
    // and its row dual is included in their reduced costs.
    return constraint.direction == BranchDirection::greater_equal_one || !coefficient;
}

}  // namespace

PricingResult CplexPricingProblem::solve(
    const GBDInstance& instance, MachineId machine, ItemId master,
    const MasterDuals& duals, const std::vector<BranchConstraint>& constraints) {
    PricingResult result;
    if (!api_) { result.status = SolveStatus::backend_unavailable; return result; }
    if (duals.item_duals.size() != instance.items.size() || duals.machine_duals.size() != instance.machines.size() ||
        duals.branch_duals.size() != constraints.size()) {
        throw std::invalid_argument("pricing dual vector dimensions do not match the GBD model");
    }
    for (const auto& constraint : constraints) {
        if (!fixed_master_allows(constraint, machine, master)) {
            result.status = SolveStatus::infeasible;
            return result;
        }
    }

    std::vector<ItemId> candidates;
    std::vector<int> variable_of_item(instance.items.size(), -1);
    for (const auto& item : instance.items) {
        if (!instance.is_compatible(master, item.id)) continue;
        variable_of_item[static_cast<std::size_t>(item.id)] = static_cast<int>(candidates.size());
        candidates.push_back(item.id);
    }
    if (variable_of_item[static_cast<std::size_t>(master)] < 0) {
        result.status = SolveStatus::infeasible;
        return result;
    }

    ProblemGuard problem(*api_, "gbd_pricing");
    const auto lp = problem.get();
    api_->check(api_->chgobjsen(api_->environment(), lp, kCplexMaximize), "set pricing objective sense");
    const auto variable_count = static_cast<int>(candidates.size());
    std::vector<double> objective(static_cast<std::size_t>(variable_count));
    std::vector<double> lower(static_cast<std::size_t>(variable_count), 0.0);
    std::vector<double> upper(static_cast<std::size_t>(variable_count), 1.0);
    std::vector<char> types(static_cast<std::size_t>(variable_count), 'B');
    for (int c = 0; c < variable_count; ++c) {
        const auto item = candidates[static_cast<std::size_t>(c)];
        objective[static_cast<std::size_t>(c)] = instance.item(item).profit - duals.item_duals[static_cast<std::size_t>(item)];
    }
    for (std::size_t r = 0; r < constraints.size(); ++r) {
        const auto& constraint = constraints[r];
        const auto dual = duals.branch_duals[r];
        if (constraint.variable.rule == BranchingRuleKind::item_selected ||
            (constraint.variable.rule == BranchingRuleKind::item_machine && machine == constraint.variable.machine)) {
            const auto item = constraint.variable.first_item;
            const auto variable = variable_of_item[static_cast<std::size_t>(item)];
            if (variable >= 0) objective[static_cast<std::size_t>(variable)] -= dual;
        }
    }
    api_->check(api_->newcols(api_->environment(), lp, variable_count, objective.data(), lower.data(), upper.data(),
                              types.data(), nullptr), "add pricing variables");

    std::vector<double> rhs{instance.machine(machine).capacity, 1.0};
    std::vector<char> sense{'L', 'E'};
    struct ExtraRow { char sense; double rhs; std::vector<std::pair<int, double>> coefficients; };
    std::vector<ExtraRow> extra_rows;
    for (const auto& constraint : constraints) {
        const auto direction = constraint.direction;
        const auto first = constraint.variable.first_item;
        const auto first_var = first >= 0 && static_cast<std::size_t>(first) < variable_of_item.size()
                                   ? variable_of_item[static_cast<std::size_t>(first)] : -1;
        switch (constraint.variable.rule) {
            case BranchingRuleKind::item_selected:
                if (direction == BranchDirection::less_equal_zero) {
                    extra_rows.push_back({'L', 0.0,
                                          first_var < 0 ? std::vector<std::pair<int,double>>{} : std::vector<std::pair<int,double>>{{first_var, 1.0}}});
                }
                break;
            case BranchingRuleKind::item_machine:
                if (machine == constraint.variable.machine && direction == BranchDirection::less_equal_zero) {
                    extra_rows.push_back({'L', 0.0,
                                          first_var < 0 ? std::vector<std::pair<int,double>>{} : std::vector<std::pair<int,double>>{{first_var, 1.0}}});
                }
                break;
            case BranchingRuleKind::pair_together: {
                const auto second = constraint.variable.second_item;
                const auto second_var = second >= 0 && static_cast<std::size_t>(second) < variable_of_item.size()
                                            ? variable_of_item[static_cast<std::size_t>(second)] : -1;
                std::vector<std::pair<int,double>> coefficients;
                if (first_var >= 0) coefficients.emplace_back(first_var, 1.0);
                if (second_var >= 0) coefficients.emplace_back(second_var, 1.0);
                extra_rows.push_back({direction == BranchDirection::less_equal_zero ? 'L' : 'G',
                                      direction == BranchDirection::less_equal_zero ? 1.0 : 2.0,
                                      std::move(coefficients)});
                break;
            }
            case BranchingRuleKind::master_selected:
            case BranchingRuleKind::master_machine:
                break;
        }
    }
    for (const auto& row : extra_rows) { sense.push_back(row.sense); rhs.push_back(row.rhs); }
    api_->check(api_->newrows(api_->environment(), lp, static_cast<int>(rhs.size()), rhs.data(), sense.data(), nullptr, nullptr),
                "add pricing constraints");

    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<double> values;
    for (int c = 0; c < variable_count; ++c) {
        rows.push_back(0); cols.push_back(c); values.push_back(instance.item(candidates[static_cast<std::size_t>(c)]).weight);
    }
    rows.push_back(1); cols.push_back(variable_of_item[static_cast<std::size_t>(master)]); values.push_back(1.0);
    for (std::size_t r = 0; r < extra_rows.size(); ++r) {
        for (const auto [column, value] : extra_rows[r].coefficients) {
            rows.push_back(static_cast<int>(2 + r)); cols.push_back(column); values.push_back(value);
        }
    }
    api_->check(api_->chgcoeflist(api_->environment(), lp, static_cast<int>(values.size()), rows.data(), cols.data(), values.data()),
                "set pricing matrix");
    const auto optimize_status = api_->mipopt(api_->environment(), lp);
    if (optimize_status != 0) { result.status = SolveStatus::numerical_error; return result; }
    const auto status = api_->getstat(api_->environment(), lp);
    if (status == 103 || status == 119) { result.status = SolveStatus::infeasible; return result; }
    if (status == 107 || status == 108 || status == 131 || status == 132) { result.status = SolveStatus::time_limit; return result; }
    if (status != 101 && status != 102) { result.status = SolveStatus::numerical_error; return result; }

    std::vector<double> primal(static_cast<std::size_t>(variable_count));
    api_->check(api_->getx(api_->environment(), lp, primal.data(), 0, variable_count - 1), "read pricing solution");
    BatchColumn column{-1, machine, master, {}, 0.0};
    for (int c = 0; c < variable_count; ++c) {
        if (primal[static_cast<std::size_t>(c)] <= 0.5) continue;
        const auto item = candidates[static_cast<std::size_t>(c)];
        column.items.push_back(item);
        column.benefit += instance.item(item).profit;
    }
    std::sort(column.items.begin(), column.items.end());
    result.reduced_cost = column.benefit - duals.machine_duals[static_cast<std::size_t>(machine)];
    for (const auto item : column.items) result.reduced_cost -= duals.item_duals[static_cast<std::size_t>(item)];
    for (std::size_t r = 0; r < constraints.size(); ++r) {
        result.reduced_cost -= duals.branch_duals[r] * constraints[r].coefficient(column);
    }
    result.column = std::move(column);
    result.status = SolveStatus::optimal;
    return result;
}

}  // namespace slbjnv
