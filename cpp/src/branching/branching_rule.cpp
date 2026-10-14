#include "slbjnv/branching/branching_rule.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <tuple>

namespace slbjnv {
namespace {

bool fractional(double value, double tolerance) {
    return std::abs(value - std::round(value)) > tolerance;
}

void validate_solution(const std::vector<FractionalColumnValue>& solution) {
    for (const auto& entry : solution) {
        if (entry.column == nullptr) throw std::invalid_argument("branching solution contains a null column");
        if (entry.value < -1.0e-9) throw std::invalid_argument("branching solution contains a negative value");
    }
}

template <typename Key, typename Accumulate, typename Convert>
std::vector<BranchCandidate> aggregate(const std::vector<FractionalColumnValue>& solution,
                                       double tolerance, Accumulate accumulate, Convert convert) {
    validate_solution(solution);
    std::map<Key, double> values;
    for (const auto& entry : solution) accumulate(values, *entry.column, entry.value);
    std::vector<BranchCandidate> candidates;
    for (const auto& [key, value] : values) {
        if (fractional(value, tolerance)) candidates.push_back({convert(key), value});
    }
    return candidates;
}

}  // namespace

std::vector<BranchCandidate> ItemSelectedBranchingRule::candidates(
    const std::vector<FractionalColumnValue>& solution, const NumericalTolerances& tolerances) const {
    return aggregate<ItemId>(solution, tolerances.integrality,
        [](auto& values, const BatchColumn& column, double value) {
            for (const auto item : column.items) values[item] += value;
        },
        [](ItemId item) { return BranchVariable{BranchingRuleKind::item_selected, item, -1, -1}; });
}

std::vector<BranchCandidate> MasterSelectedBranchingRule::candidates(
    const std::vector<FractionalColumnValue>& solution, const NumericalTolerances& tolerances) const {
    return aggregate<ItemId>(solution, tolerances.integrality,
        [](auto& values, const BatchColumn& column, double value) { values[column.master] += value; },
        [](ItemId master) { return BranchVariable{BranchingRuleKind::master_selected, master, -1, -1}; });
}

std::vector<BranchCandidate> ItemMachineBranchingRule::candidates(
    const std::vector<FractionalColumnValue>& solution, const NumericalTolerances& tolerances) const {
    using Key = std::pair<MachineId, ItemId>;
    return aggregate<Key>(solution, tolerances.integrality,
        [](auto& values, const BatchColumn& column, double value) {
            for (const auto item : column.items) values[{column.machine, item}] += value;
        },
        [](const Key& key) { return BranchVariable{BranchingRuleKind::item_machine, key.second, -1, key.first}; });
}

std::vector<BranchCandidate> MasterMachineBranchingRule::candidates(
    const std::vector<FractionalColumnValue>& solution, const NumericalTolerances& tolerances) const {
    using Key = std::pair<MachineId, ItemId>;
    return aggregate<Key>(solution, tolerances.integrality,
        [](auto& values, const BatchColumn& column, double value) { values[{column.machine, column.master}] += value; },
        [](const Key& key) { return BranchVariable{BranchingRuleKind::master_machine, key.second, -1, key.first}; });
}

std::vector<BranchCandidate> PairTogetherBranchingRule::candidates(
    const std::vector<FractionalColumnValue>& solution, const NumericalTolerances& tolerances) const {
    using Key = std::pair<ItemId, ItemId>;
    return aggregate<Key>(solution, tolerances.integrality,
        [](auto& values, const BatchColumn& column, double value) {
            auto items = column.items;
            std::sort(items.begin(), items.end());
            for (std::size_t first = 0; first < items.size(); ++first) {
                for (std::size_t second = first + 1; second < items.size(); ++second) {
                    values[{items[first], items[second]}] += value;
                }
            }
        },
        [](const Key& key) { return BranchVariable{BranchingRuleKind::pair_together, key.first, key.second, -1}; });
}

std::vector<BranchCandidate> generate_all_branch_candidates(
    const std::vector<FractionalColumnValue>& solution, const NumericalTolerances& tolerances) {
    ItemSelectedBranchingRule rule1;
    MasterSelectedBranchingRule rule2;
    ItemMachineBranchingRule rule3;
    MasterMachineBranchingRule rule4;
    PairTogetherBranchingRule rule5;
    std::vector<BranchCandidate> result;
    for (const BranchingRule* rule : std::vector<const BranchingRule*>{&rule1, &rule2, &rule3, &rule4, &rule5}) {
        auto generated = rule->candidates(solution, tolerances);
        result.insert(result.end(), generated.begin(), generated.end());
    }
    return result;
}

}  // namespace slbjnv
