#pragma once

#include "slbjnv/branching/branch_constraint.hpp"
#include "slbjnv/model/solver_backend.hpp"

#include <vector>

namespace slbjnv {

struct MasterDuals {
    std::vector<double> item_duals;
    std::vector<double> machine_duals;
    std::vector<double> branch_duals;
};

struct MasterSolution {
    LinearSolution lp;
    MasterDuals duals;
};

class RestrictedMasterProblem {
public:
    virtual ~RestrictedMasterProblem() = default;
    virtual bool add_column(BatchColumn column) = 0;
    [[nodiscard]] virtual MasterSolution solve_relaxation() = 0;
    [[nodiscard]] virtual const std::vector<BatchColumn>& columns() const = 0;
};

}  // namespace slbjnv
