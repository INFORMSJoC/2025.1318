#include "slbjnv/branch_and_price/slbjnv_branching_policy.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace slbjnv {

std::optional<JointBranchDecision> SLbJNVBranchingPolicy::select(
    const std::vector<const BranchNode*>& active_nodes) {
    struct Pair {
        const BranchNode* node{};
        BranchCandidate candidate;
    };
    std::vector<Pair> pairs;
    std::vector<FeatureExtractor::FeatureVector> features;
    for (const auto* node : active_nodes) {
        if (node == nullptr || !node->active || node->integral) continue;
        const auto& primal = node->relaxation.master.lp.primal;
        std::vector<FractionalColumnValue> column_solution;
        for (std::size_t column = 0; column < std::min(node->columns.size(), primal.size()); ++column) {
            if (primal[column] > tolerances_.feasibility) {
                column_solution.push_back({&node->columns[column], primal[column]});
            }
        }
        auto candidates = generate_all_branch_candidates(column_solution, tolerances_);
        for (const auto& candidate : candidates) {
            pairs.push_back({node, candidate});
            features.push_back(extractor_.extract(instance_, tree_, *node, candidate));
        }
    }
    if (pairs.empty()) return std::nullopt;

    const auto started = std::chrono::steady_clock::now();
    const auto scores = predictor_.predict_scores(features);
    inference_seconds_ += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    predicted_pairs_ += pairs.size();
    if (scores.size() != pairs.size()) {
        throw std::runtime_error("score count does not match joint candidate count");
    }
    const auto selected = std::max_element(scores.begin(), scores.end(), [](double left, double right) {
        if (!std::isfinite(left)) return std::isfinite(right);
        if (!std::isfinite(right)) return false;
        return left < right;
    });
    if (selected == scores.end() || !std::isfinite(*selected)) {
        throw std::runtime_error("SLbJNV predictor returned no finite candidate score");
    }
    const auto index = static_cast<std::size_t>(std::distance(scores.begin(), selected));
    return JointBranchDecision{pairs[index].node->id, pairs[index].candidate, *selected};
}

}  // namespace slbjnv
