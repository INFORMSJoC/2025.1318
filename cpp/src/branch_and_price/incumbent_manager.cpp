#include "slbjnv/branch_and_price/incumbent_manager.hpp"

namespace slbjnv {

bool IncumbentManager::update(NodeId node_id, double objective, double elapsed_seconds) {
    if (has_incumbent() && objective <= history_.back().objective) return false;
    history_.push_back({node_id, objective, elapsed_seconds});
    return true;
}

bool IncumbentManager::has_incumbent() const noexcept { return !history_.empty(); }
double IncumbentManager::lower_bound() const noexcept {
    return has_incumbent() ? history_.back().objective : kNegativeInfinity;
}

}  // namespace slbjnv
