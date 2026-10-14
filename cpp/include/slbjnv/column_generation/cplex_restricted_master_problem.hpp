#pragma once

#include "slbjnv/column_generation/restricted_master_problem.hpp"
#include "slbjnv/instance/gbd_instance.hpp"
#include "slbjnv/model/cplex_c_api.hpp"

#include <unordered_set>

namespace slbjnv {

class CplexRestrictedMasterProblem final : public RestrictedMasterProblem {
public:
    CplexRestrictedMasterProblem(const GBDInstance& instance,
                                 std::vector<BranchConstraint> constraints,
                                 std::shared_ptr<CplexCApi> api);

    bool add_column(BatchColumn column) override;
    [[nodiscard]] MasterSolution solve_relaxation() override;
    [[nodiscard]] const std::vector<BatchColumn>& columns() const override { return columns_; }

private:
    const GBDInstance& instance_;
    std::vector<BranchConstraint> constraints_;
    std::shared_ptr<CplexCApi> api_;
    std::vector<BatchColumn> columns_;
    std::unordered_set<std::string> signatures_;
};

[[nodiscard]] std::vector<BatchColumn> make_singleton_columns(const GBDInstance& instance);

}  // namespace slbjnv
