#include "slbjnv/column_generation/cplex_restricted_master_problem.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace slbjnv {
namespace {

SolveStatus lp_status(int status) {
    if (status == 1) return SolveStatus::optimal;
    if (status == 2) return SolveStatus::unbounded;
    if (status == 3 || status == 4) return SolveStatus::infeasible;
    if (status == 11 || status == 25) return SolveStatus::time_limit;
    return SolveStatus::numerical_error;
}

class ProblemGuard {
public:
    ProblemGuard(const CplexCApi& api, const char* name) : api_(api), problem_(api.create_problem(name)) {}
    ~ProblemGuard() { api_.free_problem(problem_); }
    [[nodiscard]] CplexProblem get() const { return problem_; }
private:
    const CplexCApi& api_;
    CplexProblem problem_{};
};

}  // namespace

CplexRestrictedMasterProblem::CplexRestrictedMasterProblem(
    const GBDInstance& instance, std::vector<BranchConstraint> constraints,
    std::shared_ptr<CplexCApi> api)
    : instance_(instance), constraints_(std::move(constraints)), api_(std::move(api)) {
    if (!api_) throw std::invalid_argument("CPLEX API must not be null");
}

bool CplexRestrictedMasterProblem::add_column(BatchColumn column) {
    const auto errors = column.validate(instance_);
    if (!errors.empty()) throw std::invalid_argument("attempted to add an invalid batch column: " + errors.front());
    const auto signature = column.signature();
    if (!signatures_.insert(signature).second) return false;
    column.id = static_cast<ColumnId>(columns_.size());
    columns_.push_back(std::move(column));
    return true;
}

MasterSolution CplexRestrictedMasterProblem::solve_relaxation() {
    MasterSolution solution;
    if (columns_.empty()) {
        solution.lp.status = SolveStatus::infeasible;
        return solution;
    }

    ProblemGuard problem(*api_, "gbd_rmp");
    const auto lp = problem.get();
    api_->check(api_->chgobjsen(api_->environment(), lp, kCplexMaximize), "set RMP objective sense");

    const auto item_count = static_cast<int>(instance_.items.size());
    const auto machine_count = static_cast<int>(instance_.machines.size());
    const auto branch_count = static_cast<int>(constraints_.size());
    const auto row_count = item_count + machine_count + branch_count;
    std::vector<double> rhs(static_cast<std::size_t>(row_count), 1.0);
    std::vector<char> sense(static_cast<std::size_t>(row_count), 'L');
    for (int r = 0; r < branch_count; ++r) {
        const auto& constraint = constraints_[static_cast<std::size_t>(r)];
        rhs[static_cast<std::size_t>(item_count + machine_count + r)] =
            constraint.direction == BranchDirection::less_equal_zero ? 0.0 : 1.0;
        sense[static_cast<std::size_t>(item_count + machine_count + r)] =
            constraint.direction == BranchDirection::less_equal_zero ? 'L' : 'G';
    }
    api_->check(api_->newrows(api_->environment(), lp, row_count, rhs.data(), sense.data(), nullptr, nullptr),
                "add RMP rows");

    const auto column_count = static_cast<int>(columns_.size());
    std::vector<double> objective(static_cast<std::size_t>(column_count));
    std::vector<double> lower(static_cast<std::size_t>(column_count), 0.0);
    std::vector<double> upper(static_cast<std::size_t>(column_count), 1.0);
    for (int c = 0; c < column_count; ++c) objective[static_cast<std::size_t>(c)] = columns_[static_cast<std::size_t>(c)].benefit;
    api_->check(api_->newcols(api_->environment(), lp, column_count, objective.data(), lower.data(), upper.data(), nullptr, nullptr),
                "add RMP columns");

    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<double> values;
    for (int c = 0; c < column_count; ++c) {
        const auto& column = columns_[static_cast<std::size_t>(c)];
        for (const auto item : column.items) {
            rows.push_back(item); cols.push_back(c); values.push_back(1.0);
        }
        rows.push_back(item_count + column.machine); cols.push_back(c); values.push_back(1.0);
        for (int r = 0; r < branch_count; ++r) {
            const auto coefficient = constraints_[static_cast<std::size_t>(r)].coefficient(column);
            if (std::abs(coefficient) <= 1.0e-15) continue;
            rows.push_back(item_count + machine_count + r); cols.push_back(c); values.push_back(coefficient);
        }
    }
    api_->check(api_->chgcoeflist(api_->environment(), lp, static_cast<int>(values.size()),
                                  rows.data(), cols.data(), values.data()), "set RMP matrix");
    const auto optimize_status = api_->lpopt(api_->environment(), lp);
    if (optimize_status != 0) {
        solution.lp.status = SolveStatus::numerical_error;
        return solution;
    }
    solution.lp.status = lp_status(api_->getstat(api_->environment(), lp));
    if (solution.lp.status != SolveStatus::optimal) return solution;

    api_->check(api_->getobjval(api_->environment(), lp, &solution.lp.objective), "read RMP objective");
    solution.lp.primal.resize(columns_.size());
    api_->check(api_->getx(api_->environment(), lp, solution.lp.primal.data(), 0, column_count - 1), "read RMP primal solution");
    std::vector<double> dual(static_cast<std::size_t>(row_count));
    api_->check(api_->getpi(api_->environment(), lp, dual.data(), 0, row_count - 1), "read RMP dual solution");
    solution.lp.dual = dual;
    solution.duals.item_duals.assign(dual.begin(), dual.begin() + item_count);
    solution.duals.machine_duals.assign(dual.begin() + item_count, dual.begin() + item_count + machine_count);
    solution.duals.branch_duals.assign(dual.begin() + item_count + machine_count, dual.end());
    return solution;
}

std::vector<BatchColumn> make_singleton_columns(const GBDInstance& instance) {
    std::vector<BatchColumn> columns;
    for (const auto& machine : instance.machines) {
        for (const auto& item : instance.items) {
            if (item.weight > machine.capacity + 1.0e-8) continue;
            columns.push_back({-1, machine.id, item.id, {item.id}, item.profit});
        }
    }
    return columns;
}

}  // namespace slbjnv
