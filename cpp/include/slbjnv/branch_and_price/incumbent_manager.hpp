#pragma once

#include "slbjnv/common/types.hpp"

#include <vector>

namespace slbjnv {

struct IncumbentSnapshot {
    NodeId node_id{kNoNode};
    double objective{kNegativeInfinity};
    double elapsed_seconds{};
};

class IncumbentManager {
public:
    bool update(NodeId node_id, double objective, double elapsed_seconds);
    [[nodiscard]] bool has_incumbent() const noexcept;
    [[nodiscard]] double lower_bound() const noexcept;
    [[nodiscard]] const std::vector<IncumbentSnapshot>& history() const noexcept { return history_; }

private:
    std::vector<IncumbentSnapshot> history_;
};

}  // namespace slbjnv
