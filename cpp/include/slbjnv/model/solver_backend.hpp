#pragma once

#include "slbjnv/common/types.hpp"

#include <memory>
#include <string>
#include <vector>

namespace slbjnv {

struct LinearSolution {
    SolveStatus status{SolveStatus::not_solved};
    double objective{kNegativeInfinity};
    std::vector<double> primal;
    std::vector<double> dual;
};

class SolverBackend {
public:
    virtual ~SolverBackend() = default;
    [[nodiscard]] virtual std::string name() const = 0;
};

using SolverBackendPtr = std::shared_ptr<SolverBackend>;

}  // namespace slbjnv
