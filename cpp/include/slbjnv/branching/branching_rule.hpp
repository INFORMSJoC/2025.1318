#pragma once

#include "slbjnv/branching/branch_constraint.hpp"

#include <vector>

namespace slbjnv {

struct FractionalColumnValue {
    const BatchColumn* column{};
    double value{};
};

struct BranchCandidate {
    BranchVariable variable;
    double lp_value{};
};

class BranchingRule {
public:
    virtual ~BranchingRule() = default;
    [[nodiscard]] virtual BranchingRuleKind kind() const = 0;
    [[nodiscard]] virtual std::vector<BranchCandidate> candidates(
        const std::vector<FractionalColumnValue>& solution,
        const NumericalTolerances& tolerances) const = 0;
};

class ItemSelectedBranchingRule final : public BranchingRule {
public:
    [[nodiscard]] BranchingRuleKind kind() const override { return BranchingRuleKind::item_selected; }
    [[nodiscard]] std::vector<BranchCandidate> candidates(
        const std::vector<FractionalColumnValue>& solution,
        const NumericalTolerances& tolerances) const override;
};

class MasterSelectedBranchingRule final : public BranchingRule {
public:
    [[nodiscard]] BranchingRuleKind kind() const override { return BranchingRuleKind::master_selected; }
    [[nodiscard]] std::vector<BranchCandidate> candidates(
        const std::vector<FractionalColumnValue>& solution,
        const NumericalTolerances& tolerances) const override;
};

class ItemMachineBranchingRule final : public BranchingRule {
public:
    [[nodiscard]] BranchingRuleKind kind() const override { return BranchingRuleKind::item_machine; }
    [[nodiscard]] std::vector<BranchCandidate> candidates(
        const std::vector<FractionalColumnValue>& solution,
        const NumericalTolerances& tolerances) const override;
};

class MasterMachineBranchingRule final : public BranchingRule {
public:
    [[nodiscard]] BranchingRuleKind kind() const override { return BranchingRuleKind::master_machine; }
    [[nodiscard]] std::vector<BranchCandidate> candidates(
        const std::vector<FractionalColumnValue>& solution,
        const NumericalTolerances& tolerances) const override;
};

class PairTogetherBranchingRule final : public BranchingRule {
public:
    [[nodiscard]] BranchingRuleKind kind() const override { return BranchingRuleKind::pair_together; }
    [[nodiscard]] std::vector<BranchCandidate> candidates(
        const std::vector<FractionalColumnValue>& solution,
        const NumericalTolerances& tolerances) const override;
};

[[nodiscard]] std::vector<BranchCandidate> generate_all_branch_candidates(
    const std::vector<FractionalColumnValue>& solution,
    const NumericalTolerances& tolerances);

}  // namespace slbjnv
