#pragma once

#include "slbjnv/branch_and_price/branching_policy.hpp"
#include "slbjnv/branch_and_price/search_tree.hpp"
#include "slbjnv/features/feature_extractor.hpp"
#include "slbjnv/instance/gbd_instance.hpp"
#include "slbjnv/python_interface/predictor.hpp"

#include <cstddef>

namespace slbjnv {

class SLbJNVBranchingPolicy final : public BranchingPolicy {
public:
    SLbJNVBranchingPolicy(const GBDInstance& instance, const SearchTree& tree,
                          ScorePredictor& predictor, NumericalTolerances tolerances = {})
        : instance_(instance), tree_(tree), predictor_(predictor), tolerances_(tolerances) {}

    [[nodiscard]] std::optional<JointBranchDecision> select(
        const std::vector<const BranchNode*>& active_nodes) override;

    [[nodiscard]] double inference_seconds() const noexcept { return inference_seconds_; }
    [[nodiscard]] std::size_t predicted_pairs() const noexcept { return predicted_pairs_; }

private:
    const GBDInstance& instance_;
    const SearchTree& tree_;
    ScorePredictor& predictor_;
    NumericalTolerances tolerances_;
    FeatureExtractor extractor_;
    double inference_seconds_{};
    std::size_t predicted_pairs_{};
};

}  // namespace slbjnv
